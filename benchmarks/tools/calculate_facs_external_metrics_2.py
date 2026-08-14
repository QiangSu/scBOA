#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate external validation metrics against external ground-truth metadata.

Compares labels in an h5ad file against an external metadata CSV
(e.g. ground_truth_metadata.csv) and outputs a comprehensive metric summary
suitable for downstream cross-method benchmarking.

Metrics computed (per prediction column):
    Clustering-agreement metrics (permutation-invariant):
        1. ARI  (Adjusted Rand Index)
        2. NMI  (Normalized Mutual Information)
        3. AMI  (Adjusted Mutual Information)
        4. FMI  (Fowlkes-Mallows Index)
        5. Homogeneity
        6. Completeness
        7. V-measure
    Label-matching metrics:
        8.  direct_accuracy       (only meaningful when labels are name-comparable)
        9.  direct_macro_F1
        10. hungarian_accuracy    (Hungarian-mapped)
        11. hungarian_macro_F1

Outputs:
    - per-prediction metrics CSV
    - Hungarian mapping CSV
    - confusion matrix CSV
    - per-cell comparison CSV
    - final summary CSV (one row per prediction column, one file for the run)
    - ground-truth UMAP PNG/PDF and count table

Example:
    python calculate_facs_external_metrics.py \
        --h5ad /path/to/file.h5ad \
        --metadata_csv /path/to/ground_truth_metadata.csv \
        --metadata_barcode_col cell_id \
        --facs_col ground_truth_cell_type \
        --pred_cols leiden ctpt_consensus_prediction ctpt_individual_prediction \
        --method_name clustree \
        --output_dir /path/to/output_dir \
        --output_prefix pbmc_clustree
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    accuracy_score,
    f1_score,
)

from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------
# Fixed column order for the summary CSV.
# Downstream aggregation and compare_reviewer_methods.py rely on this.
# ------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "method",
    "prediction_column",
    "facs_column",
    "n_cells_evaluated",
    "n_facs_labels",
    "n_predicted_labels",
    # clustering-agreement metrics
    "ARI",
    "NMI",
    "AMI",
    "FMI",
    "homogeneity",
    "completeness",
    "V_measure",
    # label-matching metrics
    "direct_accuracy",
    "direct_macro_F1",
    "hungarian_accuracy",
    "hungarian_macro_F1",
]


def sanitize_filename(x):
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    return x


def read_metadata(metadata_csv, metadata_barcode_col):
    meta = pd.read_csv(metadata_csv)

    if metadata_barcode_col not in meta.columns:
        raise ValueError(
            f"metadata_barcode_col '{metadata_barcode_col}' not found in metadata CSV. "
            f"Available columns: {list(meta.columns)}"
        )

    meta[metadata_barcode_col] = meta[metadata_barcode_col].astype(str)
    meta = meta.set_index(metadata_barcode_col)
    return meta


def merge_metadata_into_adata_obs(adata, meta, facs_col):
    if facs_col not in meta.columns:
        raise ValueError(
            f"facs_col '{facs_col}' not found in metadata CSV. "
            f"Available metadata columns: {list(meta.columns)}"
        )

    adata.obs_names = adata.obs_names.astype(str)
    meta.index = meta.index.astype(str)

    before_n = adata.n_obs

    overlap_cols = [c for c in meta.columns if c in adata.obs.columns]
    if len(overlap_cols) > 0:
        print("\nWarning: the following metadata columns already exist in adata.obs:")
        print(overlap_cols)
        print("They will be overwritten by values from metadata_csv.")
        adata.obs = adata.obs.drop(columns=overlap_cols)

    joined_obs = adata.obs.join(meta, how="left")

    n_matched = joined_obs[facs_col].notna().sum()
    n_unmatched = before_n - n_matched

    print("\nMetadata join summary")
    print("---------------------")
    print(f"Cells in h5ad: {before_n}")
    print(f"Cells with matched external labels: {n_matched}")
    print(f"Cells without matched external labels: {n_unmatched}")

    if n_matched == 0:
        print("\nFirst 10 h5ad obs_names:")
        print(list(adata.obs_names[:10]))
        print("\nFirst 10 metadata index values:")
        print(list(meta.index[:10]))
        raise ValueError(
            "No cells matched between h5ad obs_names and metadata barcode column. "
            "Please check barcode/cell_id formatting."
        )

    adata.obs = joined_obs
    return adata


def direct_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, macro_f1


def hungarian_map_predictions(y_true, y_pred):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    true_labels = np.array(sorted(pd.unique(y_true)))
    pred_labels = np.array(sorted(pd.unique(y_pred)))

    cm = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    true_index = {lab: i for i, lab in enumerate(true_labels)}
    pred_index = {lab: j for j, lab in enumerate(pred_labels)}

    for t, p in zip(y_true, y_pred):
        cm[true_index[t], pred_index[p]] += 1

    confusion_df = pd.DataFrame(cm, index=true_labels, columns=pred_labels)
    confusion_df.index.name = "true_label"
    confusion_df.columns.name = "predicted_label"

    row_ind, col_ind = linear_sum_assignment(-cm)

    pred_to_true = {}
    mapping_rows = []

    for r, c in zip(row_ind, col_ind):
        true_lab = true_labels[r]
        pred_lab = pred_labels[c]
        matched_count = cm[r, c]

        pred_to_true[pred_lab] = true_lab
        mapping_rows.append({
            "predicted_label": pred_lab,
            "mapped_true_label": true_lab,
            "matched_count": matched_count
        })

    for pred_lab in pred_labels:
        if pred_lab not in pred_to_true:
            c = pred_index[pred_lab]
            best_true_idx = int(np.argmax(cm[:, c]))
            best_true_lab = true_labels[best_true_idx]
            matched_count = cm[best_true_idx, c]

            pred_to_true[pred_lab] = best_true_lab
            mapping_rows.append({
                "predicted_label": pred_lab,
                "mapped_true_label": best_true_lab,
                "matched_count": matched_count
            })

    mapped_pred = np.array([pred_to_true[p] for p in y_pred])

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df = mapping_df.sort_values(
        by=["matched_count", "predicted_label"],
        ascending=[False, True]
    )

    return mapped_pred, mapping_df, confusion_df


def output_ground_truth_counts_and_umap(
    adata,
    label_col,
    output_dir,
    output_prefix,
    umap_basis="X_umap",
    dpi=300,
    compute_umap_if_missing=False
):
    if label_col not in adata.obs.columns:
        raise ValueError(
            f"Ground-truth label column '{label_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    os.makedirs(output_dir, exist_ok=True)

    label_series = adata.obs[label_col].copy()
    label_series = label_series.dropna().astype(str)

    if label_series.shape[0] == 0:
        warnings.warn(
            f"No non-missing labels found in '{label_col}'. "
            "Skipping ground-truth count table and UMAP."
        )
        return None

    counts = label_series.value_counts()
    counts_df = pd.DataFrame({
        label_col: counts.index,
        "n_cells": counts.values
    })
    counts_df["fraction"] = counts_df["n_cells"] / counts_df["n_cells"].sum()
    counts_df["percent"] = counts_df["fraction"] * 100.0
    counts_df["legend_label"] = counts_df.apply(
        lambda r: f"{r[label_col]} (n={int(r['n_cells'])}, {r['percent']:.1f}%)",
        axis=1
    )

    count_file = os.path.join(
        output_dir,
        f"{output_prefix}_{sanitize_filename(label_col)}_counts.csv"
    )
    counts_df.to_csv(count_file, index=False)

    print("\nGround-truth label count summary")
    print("--------------------------------")
    print(f"Ground-truth column: {label_col}")
    print(f"Cells with labels: {label_series.shape[0]}")
    print(f"Unique labels: {counts_df.shape[0]}")
    print(counts_df[[label_col, "n_cells", "percent"]])
    print(f"Saved count table: {count_file}")

    if umap_basis not in adata.obsm.keys():
        if compute_umap_if_missing:
            print(
                f"\nUMAP basis '{umap_basis}' not found. "
                "Computing a new UMAP because --compute_umap_if_missing was provided."
            )
            tmp = adata.copy()
            try:
                if "X_pca" not in tmp.obsm.keys():
                    sc.pp.normalize_total(tmp, target_sum=1e4)
                    sc.pp.log1p(tmp)
                    n_comps = min(50, tmp.n_obs - 1, tmp.n_vars - 1)
                    sc.tl.pca(tmp, n_comps=n_comps, svd_solver="arpack")

                sc.pp.neighbors(tmp, use_rep="X_pca" if "X_pca" in tmp.obsm.keys() else None)
                sc.tl.umap(tmp)
                coords = tmp.obsm["X_umap"]
            except Exception as e:
                warnings.warn(
                    f"Failed to compute UMAP automatically. Error: {e}. Skipping."
                )
                return count_file
        else:
            warnings.warn(
                f"UMAP basis '{umap_basis}' not found. Skipping UMAP plot."
            )
            print("\nAvailable adata.obsm keys:")
            print(list(adata.obsm.keys()))
            return count_file
    else:
        coords = adata.obsm[umap_basis]

    if coords.shape[1] < 2:
        warnings.warn(
            f"UMAP basis '{umap_basis}' has fewer than 2 columns. Skipping UMAP plot."
        )
        return count_file

    all_labels = adata.obs[label_col].astype(object)
    valid_mask = (
        all_labels.notna().values
        & np.isfinite(coords[:, 0])
        & np.isfinite(coords[:, 1])
    )
    coords_valid = coords[valid_mask, :2]
    labels_valid = all_labels.iloc[np.where(valid_mask)[0]].astype(str)

    if coords_valid.shape[0] == 0:
        warnings.warn("No cells with both labels and valid UMAP. Skipping.")
        return count_file

    plot_counts = labels_valid.value_counts()
    ordered_labels = list(plot_counts.index)

    n_labels = len(ordered_labels)
    fig_width = 8.5
    fig_height = 6.8
    if n_labels > 12:
        fig_width = 10.5
    if n_labels > 24:
        fig_width = 12.5

    n_cells = coords_valid.shape[0]
    point_size = max(1.0, min(8.0, 25000.0 / max(n_cells, 1)))

    import matplotlib.pyplot as plt

    if n_labels <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n_labels)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / max(n_labels, 1)) for i in range(n_labels)]

    color_map = {lab: colors[i] for i, lab in enumerate(ordered_labels)}

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for lab in ordered_labels:
        idx = labels_valid.values == lab
        n_lab = int(idx.sum())
        pct_lab = 100.0 * n_lab / n_cells
        legend_label = f"{lab} (n={n_lab}, {pct_lab:.1f}%)"

        ax.scatter(
            coords_valid[idx, 0],
            coords_valid[idx, 1],
            s=point_size,
            c=[color_map[lab]],
            label=legend_label,
            alpha=0.85,
            linewidths=0,
            rasterized=True
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(
        f"Ground-truth labels on original UMAP\n"
        f"{label_col}: {n_labels} labels, {n_cells} cells"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_fontsize = 7 if n_labels <= 20 else 6
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        fontsize=legend_fontsize,
        markerscale=3
    )

    plt.tight_layout()

    png_file = os.path.join(output_dir, f"{output_prefix}_ground_truth_umap.png")
    pdf_file = os.path.join(output_dir, f"{output_prefix}_ground_truth_umap.pdf")
    fig.savefig(png_file, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    print("\nGround-truth UMAP output")
    print("------------------------")
    print(f"Used UMAP basis: {umap_basis}")
    print(f"Cells plotted: {n_cells}")
    print(f"Unique labels plotted: {n_labels}")
    print(f"Saved PNG: {png_file}")
    print(f"Saved PDF: {pdf_file}")

    return {
        "count_file": count_file,
        "png_file": png_file,
        "pdf_file": pdf_file
    }


def calculate_metrics(df, facs_col, pred_col, output_prefix, output_dir, method_name):
    """
    Calculate the full external-validation metric panel for one prediction column.
    """
    if facs_col not in df.columns:
        raise ValueError(
            f"FACS/external label column '{facs_col}' not found in adata.obs. "
            f"Available columns: {list(df.columns)}"
        )
    if pred_col not in df.columns:
        raise ValueError(
            f"Prediction column '{pred_col}' not found in adata.obs. "
            f"Available columns: {list(df.columns)}"
        )

    sub = df[[facs_col, pred_col]].copy().dropna()
    sub[facs_col] = sub[facs_col].astype(str)
    sub[pred_col] = sub[pred_col].astype(str)

    n_cells = sub.shape[0]
    if n_cells == 0:
        raise ValueError(
            f"No valid cells remain after dropping NA for facs_col='{facs_col}' "
            f"and pred_col='{pred_col}'."
        )

    y_true = sub[facs_col].values
    y_pred = sub[pred_col].values

    # --- Clustering-agreement metrics (permutation-invariant) ---
    ari         = adjusted_rand_score(y_true, y_pred)
    nmi         = normalized_mutual_info_score(y_true, y_pred)
    ami         = adjusted_mutual_info_score(y_true, y_pred)
    fmi         = fowlkes_mallows_score(y_true, y_pred)
    homo        = homogeneity_score(y_true, y_pred)
    compl       = completeness_score(y_true, y_pred)
    vmeas       = v_measure_score(y_true, y_pred)

    # --- Label-matching metrics ---
    direct_acc, direct_macro_f1 = direct_metrics(y_true, y_pred)

    mapped_pred, mapping_df, confusion_df = hungarian_map_predictions(y_true, y_pred)
    hungarian_acc      = accuracy_score(y_true, mapped_pred)
    hungarian_macro_f1 = f1_score(y_true, mapped_pred, average="macro", zero_division=0)

    n_true_labels = len(pd.unique(y_true))
    n_pred_labels = len(pd.unique(y_pred))

    result = {
        "method":                method_name,
        "prediction_column":     pred_col,
        "facs_column":           facs_col,
        "n_cells_evaluated":     n_cells,
        "n_facs_labels":         n_true_labels,
        "n_predicted_labels":    n_pred_labels,
        "ARI":                   ari,
        "NMI":                   nmi,
        "AMI":                   ami,
        "FMI":                   fmi,
        "homogeneity":           homo,
        "completeness":          compl,
        "V_measure":             vmeas,
        "direct_accuracy":       direct_acc,
        "direct_macro_F1":       direct_macro_f1,
        "hungarian_accuracy":    hungarian_acc,
        "hungarian_macro_F1":    hungarian_macro_f1,
    }

    safe_pred = sanitize_filename(pred_col)

    # Per-prediction-column metric CSV, in fixed column order
    result_df = pd.DataFrame([result])[SUMMARY_COLUMNS]
    result_file = os.path.join(
        output_dir,
        f"{output_prefix}_{safe_pred}_facs_external_metrics.csv"
    )
    result_df.to_csv(result_file, index=False)

    # Supporting outputs
    mapping_file = os.path.join(
        output_dir, f"{output_prefix}_{safe_pred}_hungarian_mapping.csv"
    )
    confusion_file = os.path.join(
        output_dir, f"{output_prefix}_{safe_pred}_confusion_matrix.csv"
    )
    per_cell_file = os.path.join(
        output_dir, f"{output_prefix}_{safe_pred}_per_cell_comparison.csv"
    )

    mapping_df.to_csv(mapping_file, index=False)
    confusion_df.to_csv(confusion_file)

    per_cell = sub.copy()
    per_cell[f"{pred_col}_hungarian_mapped"] = mapped_pred
    per_cell["direct_match"] = per_cell[facs_col].astype(str) == per_cell[pred_col].astype(str)
    per_cell["hungarian_match"] = (
        per_cell[facs_col].astype(str)
        == per_cell[f"{pred_col}_hungarian_mapped"].astype(str)
    )
    per_cell.to_csv(per_cell_file)

    print("\n============================================================")
    print(f"External validation for prediction column: {pred_col}")
    print(f"Method tag: {method_name}")
    print("============================================================")
    print(f"Cells evaluated: {n_cells}")
    print(f"Number of external labels: {n_true_labels}")
    print(f"Number of predicted labels: {n_pred_labels}")
    print("--- Clustering-agreement metrics ---")
    print(f"ARI:          {ari:.6f}")
    print(f"NMI:          {nmi:.6f}")
    print(f"AMI:          {ami:.6f}")
    print(f"FMI:          {fmi:.6f}")
    print(f"Homogeneity:  {homo:.6f}")
    print(f"Completeness: {compl:.6f}")
    print(f"V-measure:    {vmeas:.6f}")
    print("--- Label-matching metrics ---")
    print(f"Direct accuracy:      {direct_acc:.6f}")
    print(f"Direct macro-F1:      {direct_macro_f1:.6f}")
    print(f"Hungarian accuracy:   {hungarian_acc:.6f}")
    print(f"Hungarian macro-F1:   {hungarian_macro_f1:.6f}")
    print(f"Saved metric file:    {result_file}")
    print(f"Saved mapping file:   {mapping_file}")
    print(f"Saved confusion mat.: {confusion_file}")
    print(f"Saved per-cell file:  {per_cell_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calculate external validation metrics against external ground-truth labels."
    )

    parser.add_argument("--h5ad", required=True,
        help="Input h5ad file containing prediction columns in adata.obs")

    parser.add_argument("--metadata_csv", required=True,
        help="CSV file containing external ground-truth labels")

    parser.add_argument("--metadata_barcode_col", default="cell_id",
        help="Column in metadata CSV matching adata.obs_names")

    parser.add_argument("--facs_col", default="ground_truth_cell_type",
        help="Column in metadata CSV containing external ground-truth labels")

    parser.add_argument("--pred_cols", nargs="+", required=True,
        help="One or more prediction columns in adata.obs to evaluate")

    parser.add_argument("--method_name", default=None,
        help=("Name/tag of the resolution-selection or annotation method producing "
              "these predictions (e.g. clustree, chooseR, scBOA_GP). "
              "Recorded in the 'method' column of the summary CSV so multiple runs "
              "can be concatenated for cross-method comparison. "
              "Defaults to --output_prefix if omitted."))

    parser.add_argument("--output_prefix", default="facs_external_validation",
        help="Output file prefix")

    parser.add_argument("--output_dir", default=".",
        help="Directory where all output CSV files will be saved")

    parser.add_argument("--save_merged_h5ad", default=None,
        help=("Optional path or filename to save h5ad after merging metadata. "
              "Relative paths are resolved inside --output_dir."))

    parser.add_argument("--umap_basis", default="X_umap",
        help="UMAP basis in adata.obsm to use for the ground-truth UMAP plot.")

    parser.add_argument("--ground_truth_umap_dpi", type=int, default=300,
        help="DPI for ground-truth UMAP PNG output.")

    parser.add_argument("--skip_ground_truth_umap", action="store_true",
        help="If set, skip ground-truth count table and UMAP plot.")

    parser.add_argument("--compute_umap_if_missing", action="store_true",
        help=("If set and adata.obsm[umap_basis] is missing, compute a new UMAP. "
              "By default, missing UMAP simply skips the plot."))

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    method_name = args.method_name if args.method_name is not None else args.output_prefix

    print("\nInput settings")
    print("==============")
    print(f"h5ad:                    {args.h5ad}")
    print(f"metadata_csv:            {args.metadata_csv}")
    print(f"metadata_barcode_col:    {args.metadata_barcode_col}")
    print(f"facs_col:                {args.facs_col}")
    print(f"pred_cols:               {args.pred_cols}")
    print(f"method_name:             {method_name}")
    print(f"output_dir:              {args.output_dir}")
    print(f"output_prefix:           {args.output_prefix}")
    print(f"umap_basis:              {args.umap_basis}")
    print(f"skip_ground_truth_umap:  {args.skip_ground_truth_umap}")
    print(f"compute_umap_if_missing: {args.compute_umap_if_missing}")

    print("\nReading h5ad...")
    adata = sc.read_h5ad(args.h5ad)

    print(f"Loaded h5ad with {adata.n_obs} cells and {adata.n_vars} genes.")
    print("\nFirst 10 h5ad obs_names:")
    print(list(adata.obs_names[:10]))
    print("\nAvailable adata.obs columns:")
    print(list(adata.obs.columns))
    print("\nAvailable adata.obsm keys:")
    print(list(adata.obsm.keys()))

    print("\nReading metadata CSV...")
    meta = read_metadata(args.metadata_csv, args.metadata_barcode_col)
    print(f"Loaded metadata with {meta.shape[0]} rows and {meta.shape[1]} columns.")
    print("\nFirst 10 metadata index values:")
    print(list(meta.index[:10]))
    print("\nAvailable metadata columns:")
    print(list(meta.columns))

    print("\nMerging metadata into adata.obs...")
    adata = merge_metadata_into_adata_obs(
        adata=adata,
        meta=meta,
        facs_col=args.facs_col
    )

    if not args.skip_ground_truth_umap:
        output_ground_truth_counts_and_umap(
            adata=adata,
            label_col=args.facs_col,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            umap_basis=args.umap_basis,
            dpi=args.ground_truth_umap_dpi,
            compute_umap_if_missing=args.compute_umap_if_missing
        )

    if args.save_merged_h5ad is not None:
        save_h5ad_path = args.save_merged_h5ad
        if not os.path.isabs(save_h5ad_path):
            save_h5ad_path = os.path.join(args.output_dir, save_h5ad_path)
        adata.write_h5ad(save_h5ad_path)
        print(f"\nSaved merged h5ad to: {save_h5ad_path}")

    # Calculate metrics for each prediction column
    all_results = []
    for pred_col in args.pred_cols:
        result = calculate_metrics(
            df=adata.obs,
            facs_col=args.facs_col,
            pred_col=pred_col,
            output_prefix=args.output_prefix,
            output_dir=args.output_dir,
            method_name=method_name,
        )
        all_results.append(result)

    summary = pd.DataFrame(all_results)[SUMMARY_COLUMNS]

    summary_file = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_summary_all_prediction_columns.csv"
    )
    summary.to_csv(summary_file, index=False)

    print("\n============================================================")
    print("Final summary (fixed schema, ready for cross-method aggregation)")
    print("============================================================")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary)
    print(f"\nSaved summary to: {summary_file}")


if __name__ == "__main__":
    main()