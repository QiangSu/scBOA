#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sciaraffa2025-style parameter selector for scRNA-seq clustering (v2).

Reference:
  Sciaraffa et al. (2025) Front. Bioinform. 5:1562410.
  "Optimization of clustering parameters for single-cell RNA analysis
   using intrinsic goodness metrics."

The published paper provides no code; this script is a faithful
re-implementation of the algorithm described in the paper's methods and
supplementary text. Documented simplifications relative to the original:

  1. 20 stratified subsamples per parameter combination
     (original: 100).
  2. statsmodels.MixedLM (Gaussian LMM) instead of R robustlmm::rlmer
     (robust LMM). Sciaraffa2025 used RLMM to down-weight outlier
     subsamples; with 20 subsamples and per-celltype stratification the
     outlier influence is limited, and the marginal-mean ranking used
     to pick the optimum is insensitive to the robust vs non-robust
     choice in the regimes we tested.

Grid (6 categorical factors, 768 combinations per subsample):
  * gene_set:   [500, 1000, 2000, 4000]   (n HVGs)
  * method:     [gauss, umap]             (neighbor kernel)
  * metric:     [cosine, euclidean]
  * n_pcs:      [10, 20, 30, 50]
  * k_nn:       [10, 20, 30]
  * resolution: [0.5, 0.8, 1.0, 2.0]

Parallelisation
---------------
Subsamples are embarrassingly parallel. Each worker performs:
  - 4 HVG+scale+PCA passes (one per gene_set value)
  - 192 Leiden clusterings per PCA
  - 768 ARI evaluations total
Workers are pinned to a fixed number of BLAS threads via threadpoolctl.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


# -----------------------------------------------------------------------------
# Fixed algorithm grid (Sciaraffa2025 spec, extended with gene_set)
# -----------------------------------------------------------------------------
GRID_GENE_SET   = [500, 1000, 2000, 4000]
GRID_METHOD     = ["gauss", "umap"]
GRID_METRIC     = ["cosine", "euclidean"]
GRID_N_PCS      = [10, 20, 30, 50]
GRID_K_NN       = [10, 20, 30]
GRID_RESOLUTION = [0.5, 0.8, 1.0, 2.0]


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def load_10x(matrix_dir: str) -> ad.AnnData:
    matrix_dir = str(matrix_dir)
    print(f"[load] Reading 10x matrix from {matrix_dir}")
    adata = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()
    print(f"[load] AnnData shape: {adata.shape}")
    return adata


def load_ground_truth(adata: ad.AnnData,
                      gt_csv: str,
                      gt_barcode_col: str,
                      gt_label_col: str) -> ad.AnnData:
    print(f"[load] Reading ground-truth CSV: {gt_csv}")
    df = pd.read_csv(gt_csv)
    if gt_barcode_col not in df.columns:
        raise ValueError(f"barcode column '{gt_barcode_col}' not in {gt_csv}")
    if gt_label_col not in df.columns:
        raise ValueError(f"label column '{gt_label_col}' not in {gt_csv}")
    df = df.set_index(gt_barcode_col)
    common = adata.obs_names.intersection(df.index)
    print(f"[load] Cells with ground-truth label: {len(common)} / {adata.n_obs}")
    adata = adata[adata.obs_names.isin(common), :].copy()
    adata.obs["ground_truth"] = df.loc[adata.obs_names, gt_label_col].values
    keep = adata.obs["ground_truth"].notna() & (adata.obs["ground_truth"] != "")
    adata = adata[keep, :].copy()
    print(f"[load] After label filtering: {adata.shape} "
          f"({adata.obs['ground_truth'].nunique()} classes)")
    return adata


# -----------------------------------------------------------------------------
# Preprocessing: normalize + log1p only. HVG + scale + PCA are per-subsample
# and per-gene_set to make gene_set a proper grid factor.
# -----------------------------------------------------------------------------
def preprocess_norm_log(adata: ad.AnnData) -> ad.AnnData:
    print("[prep] Filter + normalize_total + log1p (full data)")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"[prep] Post-normalize shape: {adata.shape}")
    return adata


# -----------------------------------------------------------------------------
# Stratified subsampling
# -----------------------------------------------------------------------------
def make_stratified_subsamples(adata: ad.AnnData,
                               n_subsamples: int,
                               train_frac: float,
                               seed: int) -> list[np.ndarray]:
    y = adata.obs["ground_truth"].astype(str).values
    sss = StratifiedShuffleSplit(n_splits=n_subsamples,
                                 train_size=train_frac,
                                 random_state=seed)
    subs = []
    for i, (train_idx, _) in enumerate(sss.split(np.zeros(len(y)), y)):
        subs.append(train_idx)
    print(f"[subs] Generated {len(subs)} stratified subsamples "
          f"(train_frac={train_frac})")
    return subs


# -----------------------------------------------------------------------------
# Run one Leiden combo on a prepared subsample AnnData
# -----------------------------------------------------------------------------
def run_one_combo(adata_sub: ad.AnnData,
                  method: str,
                  metric: str,
                  n_pcs: int,
                  k_nn: int,
                  resolution: float,
                  pca_key: str,
                  seed: int) -> float:
    sc.pp.neighbors(adata_sub,
                    n_neighbors=k_nn,
                    n_pcs=n_pcs,
                    use_rep=pca_key,
                    method=method,
                    metric=metric,
                    random_state=seed)
    sc.tl.leiden(adata_sub,
                 resolution=resolution,
                 random_state=seed,
                 key_added="leiden_tmp")
    return float(adjusted_rand_score(
        adata_sub.obs["ground_truth"].astype(str),
        adata_sub.obs["leiden_tmp"].astype(str)))


# -----------------------------------------------------------------------------
# Full 768-combo grid for one subsample (4 HVG passes × 192 combos each)
# -----------------------------------------------------------------------------
def run_grid_on_subsample(adata_full_normlog: ad.AnnData,
                          idx: np.ndarray,
                          subsample_id: int,
                          gene_set_grid: list[int],
                          max_pcs: int,
                          seed: int) -> list[dict]:
    rows = []
    combos_downstream = list(itertools.product(GRID_METHOD, GRID_METRIC,
                                               GRID_N_PCS, GRID_K_NN,
                                               GRID_RESOLUTION))

    # Slice this subsample from the full normalize+log1p AnnData
    adata_sub_base = adata_full_normlog[idx, :].copy()

    for gene_set in gene_set_grid:
        # Fresh copy for this gene_set (HVG subsetting + scaling mutate .X)
        adata_gs = adata_sub_base.copy()

        # HVG on the subsample. flavor='seurat' works on log-normalized data.
        try:
            sc.pp.highly_variable_genes(adata_gs,
                                        n_top_genes=gene_set,
                                        flavor="seurat",
                                        subset=True)
        except Exception as e:
            for (m, met, npc, knn, res) in combos_downstream:
                rows.append({
                    "subsample_id": subsample_id, "gene_set": gene_set,
                    "method": m, "metric": met, "n_pcs": npc,
                    "k_nn": knn, "resolution": res,
                    "ARI": np.nan, "status": f"hvg_error:{type(e).__name__}",
                })
            continue

        sc.pp.scale(adata_gs, max_value=10)
        sc.tl.pca(adata_gs, n_comps=max_pcs,
                  random_state=seed, zero_center=True)

        for (m, met, npc, knn, res) in combos_downstream:
            try:
                ari = run_one_combo(adata_gs, m, met, npc, knn, res,
                                    pca_key="X_pca", seed=seed)
                status = "ok"
            except BaseException as e:
                ari = np.nan
                status = f"error:{type(e).__name__}"
            rows.append({
                "subsample_id": subsample_id,
                "gene_set": gene_set,
                "method": m,
                "metric": met,
                "n_pcs": npc,
                "k_nn": knn,
                "resolution": res,
                "ARI": ari,
                "status": status,
            })

        del adata_gs

    return rows


def _subsample_worker(adata_full_normlog: ad.AnnData,
                      idx: np.ndarray,
                      subsample_id: int,
                      gene_set_grid: list[int],
                      max_pcs: int,
                      seed: int,
                      threads_per_worker: int) -> list[dict]:
    t_b = time.time()
    if HAS_THREADPOOLCTL:
        with threadpool_limits(limits=threads_per_worker):
            rows = run_grid_on_subsample(adata_full_normlog, idx,
                                         subsample_id, gene_set_grid,
                                         max_pcs, seed)
    else:
        rows = run_grid_on_subsample(adata_full_normlog, idx,
                                     subsample_id, gene_set_grid,
                                     max_pcs, seed)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"[grid]   subsample {subsample_id}: "
          f"{n_ok}/{len(rows)} combos ok, "
          f"elapsed {time.time() - t_b:.1f}s", flush=True)
    return rows


# -----------------------------------------------------------------------------
# MixedLM fit + best-combo selection over the full 6-factor grid
# -----------------------------------------------------------------------------
def fit_mixed_model_and_select(df_long: pd.DataFrame,
                               gene_set_grid: list[int],
                               output_dir: Path,
                               prefix: str) -> dict:
    import statsmodels.formula.api as smf

    df = df_long.dropna(subset=["ARI"]).copy()
    if len(df) < 50:
        raise RuntimeError(f"Too few valid rows for MixedLM: {len(df)}")

    factor_cols = ["gene_set", "method", "metric",
                   "n_pcs", "k_nn", "resolution"]
    for c in factor_cols:
        df[c] = df[c].astype("category")

    main = [f"C({c})" for c in factor_cols]
    inter = [f"{main[i]}:{main[j]}"
             for i in range(len(main))
             for j in range(i + 1, len(main))]
    formula = "ARI ~ " + " + ".join(main + inter)
    print(f"[LMM] Fitting MixedLM ({len(main)} mains, "
          f"{len(inter)} pairwise interactions, "
          f"n={len(df)} rows)")

    md = smf.mixedlm(formula, df, groups=df["subsample_id"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdf = md.fit(reml=False, method="lbfgs")

    with open(output_dir / f"{prefix}_sciaraffa_lmm_summary.txt", "w") as fh:
        fh.write(str(mdf.summary()))

    all_combos = list(itertools.product(gene_set_grid, GRID_METHOD, GRID_METRIC,
                                        GRID_N_PCS, GRID_K_NN,
                                        GRID_RESOLUTION))
    pred_df = pd.DataFrame(all_combos, columns=factor_cols)
    for c in factor_cols:
        pred_df[c] = pred_df[c].astype(
            pd.CategoricalDtype(categories=df[c].cat.categories))

    pred_df["marginal_ARI_pred"] = mdf.predict(pred_df)
    pred_df = pred_df.sort_values("marginal_ARI_pred",
                                  ascending=False).reset_index(drop=True)
    pred_df.to_csv(output_dir / f"{prefix}_sciaraffa_marginal_predictions.csv",
                   index=False)

    best = pred_df.iloc[0].to_dict()
    print("[LMM] Best combo (max marginal ARI):")
    for k, v in best.items():
        print(f"        {k} = {v}")
    return best


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sciaraffa2025-style parameter selector "
                    "(Path B reimplementation, 6-factor grid).")
    p.add_argument("--matrix_dir", required=True)
    p.add_argument("--gt_csv", required=True)
    p.add_argument("--gt_barcode_col", default="barcode")
    p.add_argument("--gt_label_col", default="cell_type")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prefix", default="PBMC")
    p.add_argument("--gene_set_grid", type=int, nargs="+",
                   default=[500, 1000, 2000, 4000],
                   help="HVG counts to include in the Sciaraffa grid "
                        "(default 500 1000 2000 4000).")
    p.add_argument("--n_subsamples", type=int, default=20,
                   help="Number of stratified subsamples "
                        "(paper uses 100; 20 here for feasibility).")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--n_jobs", type=int, default=-1,
                   help="Parallel worker processes. -1 = "
                        "min(n_subsamples, cpu_count).")
    p.add_argument("--threads_per_worker", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update module-level grid if user overrode it
    global GRID_GENE_SET
    GRID_GENE_SET = list(args.gene_set_grid)

    # 1. Load
    adata = load_10x(args.matrix_dir)
    adata = load_ground_truth(adata, args.gt_csv,
                              args.gt_barcode_col, args.gt_label_col)

    # 2. Full-data normalize + log1p only (HVG done per subsample/gene_set)
    adata = preprocess_norm_log(adata)

    # 3. Stratified subsamples
    subs = make_stratified_subsamples(adata,
                                      n_subsamples=args.n_subsamples,
                                      train_frac=args.train_frac,
                                      seed=args.seed)

    # 4. Run 768-combo grid on each subsample in parallel
    max_pcs = max(GRID_N_PCS)
    per_sub = (len(GRID_GENE_SET) * len(GRID_METHOD) * len(GRID_METRIC)
               * len(GRID_N_PCS) * len(GRID_K_NN) * len(GRID_RESOLUTION))
    n_total = args.n_subsamples * per_sub

    if args.n_jobs == -1:
        n_jobs = min(args.n_subsamples, os.cpu_count() or 1)
    else:
        n_jobs = args.n_jobs

    print(f"[grid] Running {args.n_subsamples} subsamples "
          f"x {per_sub} combos = {n_total} clusterings "
          f"across {n_jobs} worker processes "
          f"({args.threads_per_worker} thread(s) each)")
    print(f"[grid] gene_set grid: {GRID_GENE_SET}")
    if not HAS_THREADPOOLCTL and n_jobs > 1:
        print("[warn] threadpoolctl not installed; BLAS threads will not be "
              "pinned. Install with: pip install threadpoolctl", flush=True)

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_subsample_worker)(
            adata, subs[b], b, GRID_GENE_SET, max_pcs,
            args.seed, args.threads_per_worker
        )
        for b in range(len(subs))
    )
    all_rows = [row for sub in results for row in sub]

    df_long = pd.DataFrame(all_rows)
    long_path = output_dir / f"{args.prefix}_sciaraffa_grid_ari_table.csv"
    df_long.to_csv(long_path, index=False)
    print(f"[save] Long grid table -> {long_path}")

    # 5. Fit MixedLM + pick best combo
    best = fit_mixed_model_and_select(df_long, GRID_GENE_SET,
                                      output_dir, args.prefix)

    # 6. Emit selector-schema outputs
    selected_params = {
        "method_name": "Sciaraffa2025",
        "n_hvgs": int(best["gene_set"]),
        "n_pcs": int(best["n_pcs"]),
        "n_neighbors": int(best["k_nn"]),
        "resolution": float(best["resolution"]),
        "neighbor_method": str(best["method"]),
        "neighbor_metric": str(best["metric"]),
        "marginal_ARI_pred": float(best["marginal_ARI_pred"]),
        "notes": ("Path B reimplementation of Sciaraffa2025 with the full "
                  "6-factor grid (gene_set added). 20 stratified subsamples "
                  "(paper: 100); statsmodels MixedLM instead of R robustlmm."),
    }
    json_path = output_dir / f"{args.prefix}_sciaraffa_selected_params.json"
    with open(json_path, "w") as fh:
        json.dump(selected_params, fh, indent=2)
    print(f"[save] JSON  -> {json_path}")

    row = pd.DataFrame([{
        "method": "Sciaraffa2025",
        "n_hvgs": int(best["gene_set"]),
        "n_pcs": int(best["n_pcs"]),
        "n_neighbors": int(best["k_nn"]),
        "resolution": float(best["resolution"]),
        "neighbor_method": str(best["method"]),
        "neighbor_metric": str(best["metric"]),
        "marginal_ARI_pred": float(best["marginal_ARI_pred"]),
        "n_subsamples": args.n_subsamples,
        "train_frac": args.train_frac,
        "gene_set_grid": ";".join(str(x) for x in GRID_GENE_SET),
        "status": "ok",
    }])
    row_path = output_dir / f"{args.prefix}_sciaraffa_selected_row.csv"
    row.to_csv(row_path, index=False)
    print(f"[save] Row   -> {row_path}")

    print("=" * 60)
    print(f"Done in {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()