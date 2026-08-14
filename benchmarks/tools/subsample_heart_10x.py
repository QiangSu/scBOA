#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter + stratified subsample of Litvinukova heart .h5ad to ~50k cells.

Reads:
  <H5AD_PATH>  (e.g. global_raw.h5ad from Sanger)

Filters (default):
  source  == "Nuclei"
  version == "V3"
  Used    == "Yes"

Writes (into <OUTPUT_DIR>):
  10x_matrix/matrix.mtx.gz
  10x_matrix/features.tsv.gz
  10x_matrix/barcodes.tsv.gz
  ground_truth_metadata.csv
  dataset_summary.txt

The output layout is identical to what prepare_tissue_ground_truth.R produces,
so downstream scBOA and concordance scripts need no changes.
"""

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite


# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #

DEFAULT_H5AD_PATH = (
    "/home/data/qs/scRNA_simulation_data/litvinukova_heart_ground_truth/"
    "global_raw.h5ad"
)
DEFAULT_OUTPUT_DIR = (
    "/home/data/qs/scRNA_simulation_data/"
    "litvinukova_heart_ground_truth_nuclei_v3_50k"
)
DEFAULT_N_CELLS = 50_000
DEFAULT_TRUTH_COL = "cell_type"
DEFAULT_DROP_LABELS = ("NotAssigned", "doublets")

# Technical-homogeneity filters. Set to None on the CLI to disable a filter.
DEFAULT_FILTER_SOURCE = "Nuclei"
DEFAULT_FILTER_VERSION = "V3"
DEFAULT_FILTER_USED = "Yes"

DEFAULT_SEED = 0
DATASET_NAME = "litvinukova_human_heart"


# --------------------------------------------------------------------------- #
# Filtering                                                                   #
# --------------------------------------------------------------------------- #

def apply_technical_filters(adata, source, version, used, verbose=True):
    """Filter obs on paper-provided technical columns."""
    mask = np.ones(adata.n_obs, dtype=bool)
    steps = []

    if source is not None:
        if "source" not in adata.obs.columns:
            raise ValueError("Column 'source' missing from obs.")
        step_mask = adata.obs["source"].astype(str).values == source
        mask &= step_mask
        steps.append(("source", source, int(step_mask.sum())))

    if version is not None:
        if "version" not in adata.obs.columns:
            raise ValueError("Column 'version' missing from obs.")
        step_mask = adata.obs["version"].astype(str).values == version
        mask &= step_mask
        steps.append(("version", version, int(step_mask.sum())))

    if used is not None:
        if "Used" not in adata.obs.columns:
            raise ValueError("Column 'Used' missing from obs.")
        step_mask = adata.obs["Used"].astype(str).values == used
        mask &= step_mask
        steps.append(("Used", used, int(step_mask.sum())))

    if verbose:
        print("Technical filters:")
        for col, val, n in steps:
            print(f"  {col} == {val!r}: {n} cells match")
        print(f"  Joint mask kept: {int(mask.sum())} / {adata.n_obs} cells")

    return adata[mask].copy(), steps

def apply_label_filters(
    adata,
    truth_col: str,
    min_cells_per_label: int = 0,
    top_n_labels: int = 0,
    drop_labels: list = None,
) -> tuple:
    """Filter cells by label criteria: explicit drop list, min count, top-N.

    Returns
    -------
    adata : filtered AnnData
    steps : list of (description, n_cells_after) tuples for the summary file
    """
    steps = []
    labels = adata.obs[truth_col].astype(str)

    # 1. Explicit drop list
    if drop_labels:
        keep = ~labels.isin(drop_labels)
        n_before = adata.n_obs
        adata = adata[keep].copy()
        labels = adata.obs[truth_col].astype(str)
        steps.append((f"Dropped labels {drop_labels}",
                      f"{n_before} -> {adata.n_obs}"))
        print(f"  Dropped {n_before - adata.n_obs} cells with labels {drop_labels}")

    # 2. Minimum count per label
    if min_cells_per_label > 0:
        counts = labels.value_counts()
        keep_labels = counts[counts >= min_cells_per_label].index
        keep = labels.isin(keep_labels)
        n_before = adata.n_obs
        n_labels_before = labels.nunique()
        adata = adata[keep].copy()
        labels = adata.obs[truth_col].astype(str)
        steps.append((f"Kept labels with >= {min_cells_per_label} cells",
                      f"{n_before} -> {adata.n_obs} "
                      f"({n_labels_before} -> {labels.nunique()} classes)"))
        print(f"  min_cells_per_label={min_cells_per_label}: "
              f"kept {labels.nunique()}/{n_labels_before} classes, "
              f"{adata.n_obs}/{n_before} cells")

    # 3. Top-N most abundant labels
    if top_n_labels > 0:
        counts = labels.value_counts()
        keep_labels = counts.head(top_n_labels).index
        keep = labels.isin(keep_labels)
        n_before = adata.n_obs
        n_labels_before = labels.nunique()
        adata = adata[keep].copy()
        labels = adata.obs[truth_col].astype(str)
        steps.append((f"Kept top-{top_n_labels} labels by count",
                      f"{n_before} -> {adata.n_obs} "
                      f"({n_labels_before} -> {labels.nunique()} classes)"))
        print(f"  top_n_labels={top_n_labels}: kept {list(keep_labels)}")
        print(f"    -> {adata.n_obs}/{n_before} cells")

    return adata, steps
# --------------------------------------------------------------------------- #
# Stratified subsampling                                                      #
# --------------------------------------------------------------------------- #

def stratified_indices(labels: pd.Series, n_target: int, seed: int) -> np.ndarray:
    """Proportional per-class subsample. Small classes are kept in full."""
    rng = np.random.default_rng(seed)
    counts = labels.value_counts()
    total = int(counts.sum())
    if n_target >= total:
        print("  Target >= population; returning all indices.")
        return np.arange(total)

    quotas_float = counts * (n_target / total)
    quotas = np.floor(quotas_float).astype(int)
    remainder = n_target - int(quotas.sum())
    if remainder > 0:
        frac = (quotas_float - quotas).sort_values(ascending=False)
        for cls in frac.index[:remainder]:
            quotas[cls] += 1

    picked = []
    label_arr = labels.to_numpy()
    for cls, quota in quotas.items():
        cls_idx = np.where(label_arr == cls)[0]
        take = min(int(quota), cls_idx.size)
        chosen = rng.choice(cls_idx, size=take, replace=False)
        picked.append(chosen)
    return np.sort(np.concatenate(picked))


# --------------------------------------------------------------------------- #
# Writers                                                                     #
# --------------------------------------------------------------------------- #

def write_10x(adata, out_10x_dir: Path) -> None:
    out_10x_dir.mkdir(parents=True, exist_ok=True)

    # matrix.mtx.gz  (genes x cells, integer counts)
    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    X_gc = X.T.tocoo()
    if not np.issubdtype(X_gc.dtype, np.integer):
        X_int = X_gc.copy()
        X_int.data = np.rint(X_int.data).astype(np.int32)
    else:
        X_int = X_gc

    mtx_path = out_10x_dir / "matrix.mtx"
    mtx_gz_path = out_10x_dir / "matrix.mtx.gz"
    print(f"  Writing {mtx_gz_path} ...")
    mmwrite(str(mtx_path), X_int, field="integer", symmetry="general")
    with open(mtx_path, "rb") as fin, gzip.open(mtx_gz_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    mtx_path.unlink()

    # features.tsv.gz  (gene_id  gene_symbol  "Gene Expression")
    dst_features = out_10x_dir / "features.tsv.gz"
    print(f"  Writing {dst_features} ...")
    if "gene_ids" in adata.var.columns:
        gene_ids = adata.var["gene_ids"].astype(str).tolist()
    elif "gene_id" in adata.var.columns:
        gene_ids = adata.var["gene_id"].astype(str).tolist()
    else:
        # Fall back: reuse symbol as id (downstream tools accept this)
        gene_ids = adata.var_names.astype(str).tolist()
    gene_symbols = adata.var_names.astype(str).tolist()
    with gzip.open(dst_features, "wt") as f:
        for gid, gsym in zip(gene_ids, gene_symbols):
            f.write(f"{gid}\t{gsym}\tGene Expression\n")

    # barcodes.tsv.gz
    dst_barcodes = out_10x_dir / "barcodes.tsv.gz"
    print(f"  Writing {dst_barcodes} ...")
    with gzip.open(dst_barcodes, "wt") as f:
        for bc in adata.obs_names.astype(str):
            f.write(f"{bc}\n")


def write_metadata_csv(adata, truth_col: str, out_path: Path) -> None:
    """Write ground_truth_metadata.csv matching prepare_tissue_ground_truth.R schema,
    plus HLCA-style multi-level annotation columns for granularity analysis."""
    df = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "ground_truth_cell_type": adata.obs[truth_col].astype(str).values,
        "dataset": DATASET_NAME,
        "external_label_column": truth_col,
    })

    # --- Multi-level annotation columns (for granularity concordance) ---
    # HLCA-style names so the concordance script picks them up automatically.
    if "cell_type" in adata.obs.columns:
        df["ann_level_1"] = adata.obs["cell_type"].astype(str).values      # 13 broad classes
    if "cell_states" in adata.obs.columns:
        df["ann_finest_level"] = adata.obs["cell_states"].astype(str).values  # 67 fine classes

    # --- Technical provenance columns (unchanged) ---
    for col in ("source", "version", "Used", "donor", "region", "sample"):
        if col in adata.obs.columns:
            df[col] = adata.obs[col].astype(str).values

    df.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    print(f"  Levels exported: "
          f"ann_level_1={df['ann_level_1'].nunique() if 'ann_level_1' in df else 0}, "
          f"ann_finest_level={df['ann_finest_level'].nunique() if 'ann_finest_level' in df else 0}")


def write_summary(
    out_dir: Path,
    n_cells: int,
    n_genes: int,
    label_counts: pd.Series,
    truth_col: str,
    seed: int,
    source_h5ad: Path,
    filter_steps,
) -> None:
    path = out_dir / "dataset_summary.txt"
    with open(path, "w") as f:
        f.write("Stratified subsample of Litvinukova human heart .h5ad\n")
        f.write(f"Source .h5ad: {source_h5ad}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Truth column: {truth_col}\n")
        f.write("Technical filters applied:\n")
        for col, val, n in filter_steps:
            f.write(f"  {col} == {val}   ({n} cells matched this filter individually)\n")
        f.write(f"Number of cells (after filter + subsample): {n_cells}\n")
        f.write(f"Number of genes/features: {n_genes}\n")
        f.write(f"Number of external cell types: {label_counts.size}\n\n")
        f.write("External cell-type counts:\n")
        f.write(label_counts.to_string())
        f.write("\n")
    print(f"  Wrote {path}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad_path", default=DEFAULT_H5AD_PATH,
                   help="Raw .h5ad from Sanger heart cell atlas.")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n_cells", type=int, default=DEFAULT_N_CELLS)
    p.add_argument("--truth_col", default=DEFAULT_TRUTH_COL)

    # --- Technical filters ---
    p.add_argument("--filter_source", default=DEFAULT_FILTER_SOURCE,
                   help="Filter obs['source'] to this value. Pass '' to disable.")
    p.add_argument("--filter_version", default=DEFAULT_FILTER_VERSION,
                   help="Filter obs['version'] to this value. Pass '' to disable.")
    p.add_argument("--filter_used", default=DEFAULT_FILTER_USED,
                   help="Filter obs['Used'] to this value. Pass '' to disable.")

    # --- Label-based filters ---
    p.add_argument("--drop_labels", nargs="*", default=list(DEFAULT_DROP_LABELS),
                   help="Explicit list of cell-type labels to drop "
                        "(e.g. 'doublets NotAssigned').")
    p.add_argument("--keep_all_labels", action="store_true",
                   help="Do not drop any labels; overrides --drop_labels.")
    p.add_argument("--min_cells_per_label", type=int, default=0,
                   help="Drop cell types with fewer than N cells (applied AFTER "
                        "technical filters, BEFORE stratified subsampling). "
                        "0 = no filter.")
    p.add_argument("--top_n_labels", type=int, default=0,
                   help="Keep only the top N most abundant cell types (applied "
                        "AFTER --min_cells_per_label). 0 = keep all.")

    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _norm(v):
    """Treat empty string as 'disable filter'."""
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def main():
    args = parse_args()

    h5ad_path = Path(args.h5ad_path)
    output_dir = Path(args.output_dir)
    if not h5ad_path.exists():
        sys.exit(f"Missing .h5ad: {h5ad_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load raw .h5ad ---
    print(f"Reading .h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    if args.truth_col not in adata.obs.columns:
        sys.exit(f"Truth column '{args.truth_col}' not in obs. "
                 f"Available: {list(adata.obs.columns)}")

    # --- 2. Technical filters (Nuclei / V3 / Used=Yes by default) ---
    adata, tech_steps = apply_technical_filters(
        adata,
        source=_norm(args.filter_source),
        version=_norm(args.filter_version),
        used=_norm(args.filter_used),
    )
    print(f"After technical filters: {adata.n_obs} cells")

    # --- 3. Label-based filters (NEW: replaces the old drop-labels block) ---
    drop_list = [] if args.keep_all_labels else args.drop_labels
    adata, label_steps = apply_label_filters(
        adata,
        truth_col=args.truth_col,
        min_cells_per_label=args.min_cells_per_label,
        top_n_labels=args.top_n_labels,
        drop_labels=drop_list,
    )
    print(f"After label filters: {adata.n_obs} cells")

    if adata.n_obs == 0:
        sys.exit("No cells left after filtering.")

    # --- 4. Stratified subsample ---
    print(f"Stratified subsampling to ~{args.n_cells} cells ...")
    idx = stratified_indices(
        labels=adata.obs[args.truth_col].reset_index(drop=True),
        n_target=args.n_cells,
        seed=args.seed,
    )
    adata_sub = adata[idx].copy()
    print(f"  Subsampled: {adata_sub.n_obs} cells")

    # --- 5. Write outputs ---
    out_10x = output_dir / "10x_matrix"
    write_10x(adata_sub, out_10x)

    meta_out_path = output_dir / "ground_truth_metadata.csv"
    write_metadata_csv(adata_sub, args.truth_col, meta_out_path)

    label_counts = adata_sub.obs[args.truth_col].value_counts()

    # tech_steps is a list of (col, val, n) tuples; label_steps is (desc, delta).
    # Normalize both into the (label, value, n) shape write_summary expects.
    combined_steps = list(tech_steps) + [
        (desc, "-", info) for desc, info in label_steps
    ]

    write_summary(
        out_dir=output_dir,
        n_cells=adata_sub.n_obs,
        n_genes=adata_sub.n_vars,
        label_counts=label_counts,
        truth_col=args.truth_col,
        seed=args.seed,
        source_h5ad=h5ad_path,
        filter_steps=combined_steps,
    )

    print("\nDone.")
    print(f"Output directory: {output_dir}")
    print(label_counts)


if __name__ == "__main__":
    main()