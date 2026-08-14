#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scLENS wrapper for the scBOA baseline benchmark.

scLENS (Kim et al. Nat. Commun. 2024) is a complete pipeline that:
  - Uses its own QC + L1/log/Z-score/L2 normalization (no HVG selection)
  - Automatically detects "robust signal components" (n_pcs equivalent) via
    Tracy-Widom threshold + perturbation robustness
  - Runs Leiden clustering on the robust PC embedding

For the benchmark it therefore contributes:
  - n_hvgs      : NA (uses all QC-passed genes) -> reported as total gene count
  - n_pcs       : automatically selected by scLENS
  - k_nn        : user-specified (scLENS itself does not choose k)
  - resolution  : scanned over user-provided grid, best selected by
                  silhouette on the scLENS embedding

Clustering here uses scanpy Leiden on the scLENS embedding, which matches
the downstream engine used by every other baseline in this benchmark
(findPC, eigengap, clustree, chooseR, MultiK, scICE, ...) and avoids the
`find_clusters(random_state=...)` API mismatch with newer leidenalg.

Output JSON matches the schema used by the R baseline selectors
(findPC / eigengap / clustree / chooseR / etc.).
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io as sio
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_10x_matrix(matrix_dir):
    """Load 10x-format matrix.mtx.gz + barcodes.tsv.gz + features.tsv.gz."""
    matrix_dir = Path(matrix_dir)
    mtx = sio.mmread(matrix_dir / "matrix.mtx.gz").tocsr()  # genes x cells
    barcodes = pd.read_csv(matrix_dir / "barcodes.tsv.gz",
                           header=None, sep="\t")[0].values
    features = pd.read_csv(matrix_dir / "features.tsv.gz",
                           header=None, sep="\t")
    gene_names = features[1].values if features.shape[1] >= 2 else features[0].values

    dense = mtx.T.toarray()  # cells x genes
    df = pd.DataFrame(dense, index=barcodes, columns=gene_names)
    if not df.columns.is_unique:
        df = df.T.groupby(level=0).sum().T
    return df


def parse_grid(spec):
    """Parse '0.1,0.3,0.5' -> [0.1, 0.3, 0.5]."""
    return [float(x) for x in spec.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# scLENS runner
# ---------------------------------------------------------------------------
def run_sclens(df, resolutions, k_nn, seed, device_str, min_genes, min_cells):
    """Fit scLENS, then run scanpy Leiden on the scLENS embedding."""
    import torch
    from scLENS import scLENS

    device = torch.device(device_str) if device_str else \
             (torch.device("cuda:0") if torch.cuda.is_available()
              else torch.device("cpu"))

    print(f"[scLENS] device = {device}")
    sclens = scLENS(device=device)

    print("[scLENS] Preprocessing (L1/log/Z-score/L2)...")
    t0 = time.time()
    sclens.preprocess(df,
                      min_genes_per_cell=min_genes,
                      min_cells_per_gene=min_cells,
                      verb=True)
    print(f"[scLENS]   preprocess done in {time.time() - t0:.1f}s")

    print("[scLENS] Fitting robust signal components...")
    t0 = time.time()
    X_transform = sclens.fit_transform(plot_mp=False)
    print(f"[scLENS]   fit_transform done in {time.time() - t0:.1f}s")

    # Normalize embedding to a numpy array on CPU
    if hasattr(X_transform, "detach"):
        X_transform = X_transform.detach().cpu().numpy()
    X_transform = np.asarray(X_transform, dtype=np.float32)

    n_signal = int(X_transform.shape[1])
    n_cells_post = int(X_transform.shape[0])

    # Recover post-QC gene count (attribute name varies across scLENS versions)
    if hasattr(sclens, "normal_genes") and sclens.normal_genes is not None:
        try:
            n_genes_post = int(len(sclens.normal_genes))
        except TypeError:
            n_genes_post = int(np.asarray(sclens.normal_genes).size)
    else:
        n_genes_post = int(df.shape[1])

    # Recover post-QC cell names
    obs_names = None
    if hasattr(sclens, "obs_names") and sclens.obs_names is not None:
        obs_all = np.asarray(sclens.obs_names)
        if hasattr(sclens, "normal_cells") and sclens.normal_cells is not None:
            try:
                obs_names = obs_all[np.asarray(sclens.normal_cells)]
            except Exception:
                obs_names = obs_all
        else:
            obs_names = obs_all
    if obs_names is None or len(obs_names) != n_cells_post:
        obs_names = np.array([f"cell_{i}" for i in range(n_cells_post)])

    print(f"[scLENS]   robust signal components (n_pcs) = {n_signal}")
    print(f"[scLENS]   cells kept = {n_cells_post}, genes kept = {n_genes_post}")

    # ---- Build AnnData wrapper around the scLENS embedding -----------
    adata = ad.AnnData(X=X_transform)
    adata.obs_names = pd.Index([str(x) for x in obs_names])
    adata.obsm["X_sclens"] = X_transform

    print(f"[scLENS] Building kNN graph on scLENS embedding "
          f"(k_nn = {k_nn}) ...")
    sc.pp.neighbors(adata,
                    n_neighbors=k_nn,
                    use_rep="X_sclens",
                    random_state=seed)

    # ---- Sweep resolutions with scanpy Leiden ------------------------
    results = []
    for res in resolutions:
        key = f"leiden_res{res}"
        print(f"[scLENS]   scanpy Leiden at resolution={res} ...")
        try:
            sc.tl.leiden(adata,
                         resolution=float(res),
                         random_state=seed,
                         key_added=key)
            labels = adata.obs[key].astype(str).values
            # convert to integer codes for downstream metrics
            _, int_labels = np.unique(labels, return_inverse=True)
            n_clust = int(len(np.unique(int_labels)))
        except Exception as e:
            print(f"[scLENS]     ! Leiden failed at res={res}: {e}")
            int_labels = None
            n_clust = 0
        results.append({
            "resolution": float(res),
            "n_clusters": n_clust,
            "labels": int_labels,
        })

    return {
        "n_signal_components": n_signal,
        "n_cells_post_qc": n_cells_post,
        "n_genes_post_qc": n_genes_post,
        "embedding": X_transform,
        "cluster_results": results,
        "obs_names": obs_names,
    }


# ---------------------------------------------------------------------------
# Resolution selection: silhouette on scLENS embedding
# ---------------------------------------------------------------------------
def select_best_resolution(fit_out, silhouette_sample, seed):
    """Pick resolution with highest silhouette on scLENS embedding."""
    from sklearn.metrics import silhouette_score

    emb = fit_out["embedding"]
    n_cells = emb.shape[0]

    rng = np.random.default_rng(seed)
    if n_cells > silhouette_sample:
        idx = rng.choice(n_cells, size=silhouette_sample, replace=False)
    else:
        idx = np.arange(n_cells)
    emb_s = emb[idx]

    rows = []
    for r in fit_out["cluster_results"]:
        res = r["resolution"]
        labels = r["labels"]
        n_clust = r["n_clusters"]
        if labels is None or n_clust < 2:
            sil = np.nan
        else:
            try:
                sil = float(silhouette_score(emb_s, labels[idx]))
            except Exception as e:
                print(f"[scLENS]   silhouette failed at res={res}: {e}")
                sil = np.nan
        rows.append({
            "resolution": res,
            "n_clusters": n_clust,
            "silhouette": sil,
        })
    sweep = pd.DataFrame(rows)
    if sweep["silhouette"].notna().any():
        best_idx = sweep["silhouette"].idxmax()
        best_res = float(sweep.loc[best_idx, "resolution"])
    else:
        # fallback: median resolution
        best_res = float(sweep.loc[len(sweep) // 2, "resolution"])
    return best_res, sweep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="scLENS wrapper for scBOA baseline benchmark.")
    p.add_argument("--matrix_dir", required=True,
                   help="10x matrix directory (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prefix", default="sclens_run")
    p.add_argument("--resolutions", default="0.1,0.3,0.5,0.8,1.0,1.5,2.0",
                   help="Comma-separated resolution grid.")
    p.add_argument("--k_nn", type=int, default=20,
                   help="k for the kNN graph passed to Leiden.")
    p.add_argument("--min_genes_per_cell", type=int, default=200)
    p.add_argument("--min_cells_per_gene", type=int, default=15)
    p.add_argument("--silhouette_sample", type=int, default=2000)
    p.add_argument("--device", default=None,
                   help="'cuda:0' or 'cpu'. Default: auto.")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    resolutions = parse_grid(args.resolutions)
    print("=" * 60)
    print("scLENS baseline wrapper")
    print("=" * 60)
    print(f"matrix_dir       : {args.matrix_dir}")
    print(f"output_dir       : {args.output_dir}")
    print(f"prefix           : {args.prefix}")
    print(f"resolutions      : {resolutions}")
    print(f"k_nn             : {args.k_nn}")
    print(f"seed             : {args.seed}")
    print("-" * 60)

    # ---- Load data ----------------------------------------------------
    print("[load] Reading 10x matrix ...")
    df = load_10x_matrix(args.matrix_dir)
    print(f"[load]   raw shape (cells x genes) = {df.shape}")

    # ---- Run scLENS --------------------------------------------------
    fit_out = run_sclens(df,
                         resolutions=resolutions,
                         k_nn=args.k_nn,
                         seed=args.seed,
                         device_str=args.device,
                         min_genes=args.min_genes_per_cell,
                         min_cells=args.min_cells_per_gene)

    # ---- Select best resolution --------------------------------------
    best_res, sweep = select_best_resolution(fit_out,
                                             args.silhouette_sample,
                                             args.seed)
    n_clust_best = int(sweep.loc[sweep["resolution"] == best_res,
                                 "n_clusters"].iloc[0])
    print(f"[select] Best resolution = {best_res}  "
          f"(n_clusters = {n_clust_best})")

    # ---- Save labels for the winning resolution ----------------------
    winner = next(r for r in fit_out["cluster_results"]
                  if r["resolution"] == best_res)
    if winner["labels"] is not None:
        labels_df = pd.DataFrame({
            "cell": fit_out["obs_names"],
            "sclens_cluster": winner["labels"],
        })
        labels_path = os.path.join(
            args.output_dir,
            f"{args.prefix}_sclens_labels.csv")
        labels_df.to_csv(labels_path, index=False)
        print(f"[save] labels -> {labels_path}")

    # ---- Save resolution sweep ---------------------------------------
    sweep_path = os.path.join(
        args.output_dir,
        f"{args.prefix}_sclens_resolution_sweep.csv")
    sweep.to_csv(sweep_path, index=False)
    print(f"[save] sweep  -> {sweep_path}")

    # ---- Save selected-params JSON (schema matches R baselines) ------
    selected = {
        "method": "scLENS",
        "reference": "Kim et al. Nat. Commun. 2024",
        "n_hvgs": None,                          # scLENS does not use HVGs
        "n_hvgs_note": "scLENS uses all QC-passed genes; no HVG selection.",
        "n_genes_used": fit_out["n_genes_post_qc"],
        "n_pcs": fit_out["n_signal_components"],
        "n_pcs_source": "scLENS robust signal components (Tracy-Widom + perturbation robustness)",
        "k_nn": args.k_nn,
        "k_nn_source": "user-specified (scLENS does not select k)",
        "resolution": best_res,
        "resolution_source": "silhouette maximum on scLENS embedding across grid (scanpy Leiden)",
        "n_clusters": n_clust_best,
        "resolution_grid": resolutions,
        "seed": args.seed,
        "clustering_engine": "scanpy.tl.leiden on X_sclens",
        "status": "ok",
    }
    json_path = os.path.join(
        args.output_dir,
        f"{args.prefix}_sclens_selected_params.json")
    with open(json_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"[save] params -> {json_path}")

    # ---- Save one-row summary CSV (matches R selector layout) --------
    row = pd.DataFrame([{
        "method": "scLENS",
        "n_hvgs": np.nan,
        "n_pcs": fit_out["n_signal_components"],
        "k_nn": args.k_nn,
        "resolution": best_res,
        "n_clusters": n_clust_best,
        "status": "ok",
    }])
    row_path = os.path.join(
        args.output_dir,
        f"{args.prefix}_sclens_selected_row.csv")
    row.to_csv(row_path, index=False)
    print(f"[save] row    -> {row_path}")

    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()