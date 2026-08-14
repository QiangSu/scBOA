#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scICE-style stable-resolution selector (Python companion to the R selector
benchmark).

Kim et al. 2025 (Nat Commun) scICE identifies stable clustering resolutions by
running Leiden many times per resolution with different random seeds and
quantifying inconsistency across runs. This script re-implements the core
algorithm in Python (the official scICE package is Julia-only):

  1. For each candidate Leiden resolution, run Leiden N times with different
     random seeds on the SAME kNN graph (built once from HVG/PCA/neighbors
     matching the R pipeline).
  2. Compute pairwise clustering consistency (ARI + NMI) across the N runs.
  3. Define stability: resolutions with mean pairwise ARI >= --ari_threshold
     are considered "stable". Among stable resolutions with n_clusters <=
     --max_k, pick the FINEST (highest resolution) - this is the scICE
     selection rule (finest stable partition).
  4. Emit {prefix}_scICE_selected_params.json in the same schema as the R
     build_params() output so it merges cleanly into the main summary CSV.

Usage (run from any dir; assumes conda env r-seurat-selection is active):

  conda activate r-seurat-selection
  python scice_selector.py \
      --matrix_dir /path/to/10x_matrix \
      --output_dir /path/to/output \
      --prefix pbmc_bench \
      --n_hvgs 2000 --n_pcs 30 --n_neighbors 15 \
      --resolutions 0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.5,2.0 \
      --n_replicates 30 --ari_threshold 0.90
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependency check and auto-install (runs FIRST, before other imports)
# ---------------------------------------------------------------------------

REQUIRED = {
    # import_name : pip_package_name
    "numpy":       "numpy",
    "scipy":       "scipy",
    "pandas":      "pandas",
    "matplotlib":  "matplotlib",
    "sklearn":     "scikit-learn",
    "anndata":     "anndata",
    "scanpy":      "scanpy",
    "leidenalg":   "leidenalg",
    "igraph":      "python-igraph",
}


def ensure_dependencies():
    """Check every REQUIRED module; pip-install any that are missing."""
    print("=" * 70)
    print("[deps] Checking Python dependencies in current environment:")
    print(f"       {sys.executable}")
    print("=" * 70)

    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            importlib.import_module(mod)
            print(f"  [ok]      {mod}")
        except ImportError:
            print(f"  [MISSING] {mod}  ->  will pip install {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\n[deps] Installing: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        )
        # verify installation
        for mod in REQUIRED:
            importlib.import_module(mod)
        print("[deps] All missing packages installed successfully.")
    else:
        print("[deps] All required packages already installed.")
    print("=" * 70)


ensure_dependencies()


# ---------------------------------------------------------------------------
# Real imports (only after dependency check)
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="scICE-style stable-resolution selector (Python)."
    )
    p.add_argument("--matrix_dir", required=True,
                   help="Directory containing 10x barcodes/features/matrix files")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prefix", default="scice")

    p.add_argument("--n_hvgs", type=int, default=2000)
    p.add_argument("--n_pcs", type=int, default=30)
    p.add_argument("--n_neighbors", type=int, default=15)

    p.add_argument("--resolutions", type=str,
                   default="0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.5,2.0",
                   help="Comma-separated Leiden resolutions to test")

    p.add_argument("--n_replicates", type=int, default=30,
                   help="Leiden replicates per resolution (scICE paper uses 100; "
                        "20-30 already gives a robust stability ranking)")
    p.add_argument("--ari_threshold", type=float, default=0.90,
                   help="Mean pairwise ARI required to declare a resolution stable")

    p.add_argument("--max_k", type=int, default=50)
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--min_features", type=int, default=200)
    p.add_argument("--max_percent_mt", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing (mirrors the R Seurat pipeline)
# ---------------------------------------------------------------------------

def load_and_preprocess(args):
    print(f"\n[1] Loading 10x matrix from {args.matrix_dir}")
    adata = sc.read_10x_mtx(args.matrix_dir, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()
    print(f"    Raw: {adata.n_obs} cells x {adata.n_vars} genes")

    print("[2] QC filtering")
    sc.pp.filter_cells(adata, min_genes=args.min_features)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith(("MT-", "MT."))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs["pct_counts_mt"] <= args.max_percent_mt].copy()
    print(f"    Post-QC: {adata.n_obs} cells x {adata.n_vars} genes")

    print("[3] Normalize / HVG / scale / PCA / neighbors")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs, flavor="seurat")
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)

    n_comps = max(args.n_pcs, 50)
    n_comps = min(n_comps, min(adata.shape) - 1)
    sc.tl.pca(adata, n_comps=n_comps, random_state=args.seed)

    sc.pp.neighbors(
        adata,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
        random_state=args.seed,
    )
    return adata


# ---------------------------------------------------------------------------
# scICE core: N Leiden replicates per resolution + pairwise consistency
# ---------------------------------------------------------------------------

def run_leiden_replicates(adata, resolution, n_replicates, base_seed):
    """Run Leiden N times at a given resolution with different seeds."""
    labels_list = []
    for k in range(n_replicates):
        seed_k = base_seed + 10_000 + k
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=seed_k,
            key_added=f"_tmp_leiden_r{resolution}_s{seed_k}",
        )
        labels_list.append(
            adata.obs[f"_tmp_leiden_r{resolution}_s{seed_k}"].astype(str).values
        )
        del adata.obs[f"_tmp_leiden_r{resolution}_s{seed_k}"]
    return labels_list


def pairwise_consistency(labels_list):
    """
    Mean and std of pairwise ARI + NMI across all replicate clusterings.
    This is the scICE consistency proxy (finest partition still-stable rule).
    """
    n = len(labels_list)
    aris, nmis = [], []
    for i in range(n):
        for j in range(i + 1, n):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
            nmis.append(normalized_mutual_info_score(labels_list[i], labels_list[j]))
    return (
        float(np.mean(aris)), float(np.std(aris)),
        float(np.mean(nmis)), float(np.std(nmis)),
    )


def modal_n_clusters(labels_list):
    counts = [len(np.unique(lbl)) for lbl in labels_list]
    return int(pd.Series(counts).mode().iloc[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolutions = sorted(set(float(x) for x in args.resolutions.split(",")))

    print("=" * 70)
    print("scICE-style stable-resolution selector")
    print(f"  matrix_dir : {args.matrix_dir}")
    print(f"  output_dir : {out_dir}")
    print(f"  resolutions: {resolutions}")
    print(f"  replicates : {args.n_replicates}   ARI threshold: {args.ari_threshold}")
    print("=" * 70)

    adata = load_and_preprocess(args)

    # --- resolution sweep with N replicates each ---
    print(f"\n[4] Running {args.n_replicates} Leiden replicates at each of "
          f"{len(resolutions)} resolutions...")
    rows = []
    for r in resolutions:
        print(f"    resolution {r:.3f} ...", end="", flush=True)
        labels_list = run_leiden_replicates(
            adata, resolution=r,
            n_replicates=args.n_replicates,
            base_seed=args.seed,
        )
        mean_ari, sd_ari, mean_nmi, sd_nmi = pairwise_consistency(labels_list)
        n_k = modal_n_clusters(labels_list)
        rows.append(dict(
            resolution=r,
            modal_n_clusters=n_k,
            mean_pairwise_ari=mean_ari,
            sd_pairwise_ari=sd_ari,
            mean_pairwise_nmi=mean_nmi,
            sd_pairwise_nmi=sd_nmi,
            inconsistency_coef=(1.0 / mean_ari) if mean_ari > 0 else np.inf,
            is_stable=bool(mean_ari >= args.ari_threshold and n_k <= args.max_k),
        ))
        print(f"  mean ARI = {mean_ari:.3f}   n_k = {n_k}   "
              f"stable = {rows[-1]['is_stable']}")

    stability_df = pd.DataFrame(rows)
    stab_csv = out_dir / f"{args.prefix}_scICE_stability.csv"
    stability_df.to_csv(stab_csv, index=False)
    print(f"\n[5] Stability table saved: {stab_csv}")

    # --- selection rule: finest stable resolution (largest r with is_stable) ---
    stable = stability_df[stability_df["is_stable"]]
    if len(stable) > 0:
        selected = stable.loc[stable["resolution"].idxmax()]
        reason = (f"Finest resolution with mean pairwise ARI "
                  f">= {args.ari_threshold} and n_clusters <= {args.max_k}")
        status = "ok"
    else:
        # fallback: resolution with highest ARI subject to n_k <= max_k
        cand = stability_df[stability_df["modal_n_clusters"] <= args.max_k]
        if len(cand) == 0:
            cand = stability_df
        selected = cand.loc[cand["mean_pairwise_ari"].idxmax()]
        reason = ("Fallback: no resolution met stability threshold; "
                  "picked resolution with highest mean pairwise ARI")
        status = "ok_fallback"

    print("\n[6] Selection:")
    print(f"    resolution = {selected['resolution']}")
    print(f"    n_clusters = {int(selected['modal_n_clusters'])}")
    print(f"    mean ARI   = {selected['mean_pairwise_ari']:.3f}")
    print(f"    reason     = {reason}")

    # --- emit JSON in the same schema as R build_params() ---
    params = {
        "method": "scICE",
        "n_hvgs": args.n_hvgs,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "resolution": float(selected["resolution"]),
        "selected_n_clusters": int(selected["modal_n_clusters"]),
        "reason": reason,
        # scICE-specific diagnostics (extra):
        "mean_pairwise_ari": float(selected["mean_pairwise_ari"]),
        "sd_pairwise_ari":  float(selected["sd_pairwise_ari"]),
        "mean_pairwise_nmi": float(selected["mean_pairwise_nmi"]),
        "inconsistency_coef": float(selected["inconsistency_coef"]),
        "n_replicates": args.n_replicates,
        "ari_threshold": args.ari_threshold,
    }
    json_path = out_dir / f"{args.prefix}_scICE_selected_params.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"    JSON saved: {json_path}")

    # --- one-row summary CSV, compatible with the R selected_resolution_summary
    summary_row = pd.DataFrame([{
        "method": "scICE",
        "selected_resolution": float(selected["resolution"]),
        "selected_n_clusters": int(selected["modal_n_clusters"]),
        "selected_n_hvgs": args.n_hvgs,
        "selected_n_pcs": args.n_pcs,
        "selected_n_neighbors": args.n_neighbors,
        "reason": reason,
        "status": status,
    }])
    summary_csv = out_dir / f"{args.prefix}_scICE_selected_row.csv"
    summary_row.to_csv(summary_csv, index=False)
    print(f"    Summary row saved: {summary_csv}")

    # --- diagnostic plots ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(stability_df["resolution"],
                   stability_df["mean_pairwise_ari"],
                   yerr=stability_df["sd_pairwise_ari"],
                   marker="o", capsize=3, color="steelblue")
    ax[0].axhline(args.ari_threshold, ls="--", color="red",
                  label=f"ARI threshold = {args.ari_threshold}")
    ax[0].axvline(selected["resolution"], ls=":", color="black",
                  label=f"selected r = {selected['resolution']}")
    ax[0].set_xlabel("Leiden resolution")
    ax[0].set_ylabel("Mean pairwise ARI across replicates")
    ax[0].set_title(f"scICE stability (n_reps = {args.n_replicates})")
    ax[0].legend(fontsize=8)

    ax[1].plot(stability_df["resolution"], stability_df["modal_n_clusters"],
               marker="o", color="darkgreen")
    ax[1].axvline(selected["resolution"], ls=":", color="black")
    ax[1].set_xlabel("Leiden resolution")
    ax[1].set_ylabel("Modal number of clusters")
    ax[1].set_title("Cluster count across replicates")

    plt.tight_layout()
    fig_path = out_dir / f"{args.prefix}_scICE_stability.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"    Diagnostic plot: {fig_path}")

    print("\n" + "=" * 70)
    print("scICE selector finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()