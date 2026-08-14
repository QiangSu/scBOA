#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot raw and granularity-adjusted scBOA objective-to-ground-truth heatmaps.

This script uses:
  1. spearman_internal_vs_facs.csv
  2. spearman_internal_vs_facs_partial_controlling_cluster_count_and_resolution.csv

It produces a two-panel heatmap:
  left  = raw Spearman correlation
  right = partial Spearman correlation after jointly controlling for
          n_leiden_clusters and Leiden resolution.

Example:
  python plot_raw_partial_pbmc_heatmap.py \
      --input-dir ./ \
      --output-prefix figures/pbmc_raw_partial_objective_heatmap
"""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


EXTERNAL_METRIC_ORDER = [
    "external_leiden_ari",
    "external_leiden_nmi",
    "external_homogeneity",
    "external_completeness",
    "external_celltypist_broad_accuracy",
    "external_celltypist_broad_macro_f1",
]
INTERNAL_METRICS_EXCLUDE = {
    "mean_confidence",
    "mean_confidence_score",
    "avg_confidence",
    "average_confidence",
    "celltypist_mean_confidence",
}
EXTERNAL_LABELS = {
    "external_leiden_ari": "Leiden ARI",
    "external_leiden_nmi": "Leiden NMI",
    "external_homogeneity": "Homogeneity",
    "external_completeness": "Completeness",
    "external_celltypist_broad_accuracy": "CellTypist\nbroad accuracy",
    "external_celltypist_broad_macro_f1": "CellTypist\nbroad macro-F1",
}

INTERNAL_LABELS = {
    "cas": "CAS",
    "CAS": "CAS",
    "mcs": "MCS",
    "MCS": "MCS",
    "mps": "MPS",
    "MPS": "MPS",
    "mps_f1": "MPS/F1",
    "MPS_F1": "MPS/F1",
    "f1": "F1",
    "F1": "F1",
    "final_objective": "Final objective",
    "objective": "Final objective",
    "objective_score": "Final objective",
    "final_score": "Final objective",
    "composite_objective": "Composite objective",
    "scboa_objective": "scBOA objective",
    "yield_score_target": "Yield score target",
    "balanced_score_gmean": "Balanced score gmean",
    "mean_cluster_score": "Mean cluster score",
    "mean_prediction_score": "Mean prediction score",
    "mean_confidence_score": "Mean confidence score",
}


def find_column(df, candidates, required=True, table_name="table"):
    """Return the first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"Could not find any of columns {candidates} in {table_name}. "
            f"Available columns are: {list(df.columns)}"
        )

    return None


def load_correlation_matrix(csv_path, mode="raw"):
    """
    Load a long-format correlation CSV and return a matrix:

        rows    = internal scBOA metrics
        columns = external PBMC ground-truth metrics
        values  = Spearman rho or partial Spearman rho
    """

    df = pd.read_csv(csv_path)

    internal_col = find_column(
        df,
        [
            "internal_metric",
            "objective_metric",
            "internal_score",
            "score_metric",
            "metric",
        ],
        table_name=str(csv_path),
    )

    external_col = find_column(
        df,
        [
            "external_metric",
            "external_facs_metric",
            "facs_metric",
            "ground_truth_metric",
            "external_score",
            "target_metric",
        ],
        table_name=str(csv_path),
    )

    if mode == "raw":
        rho_col = find_column(
            df,
            [
                "spearman_rho",
                "raw_spearman_rho",
                "rho",
                "spearman_r",
                "correlation",
            ],
            table_name=str(csv_path),
        )
    else:
        rho_col = find_column(
            df,
            [
                "partial_spearman_rho",
                "partial_rho",
                "spearman_partial_rho",
                "partial_correlation",
                "rho",
                "spearman_rho",
            ],
            table_name=str(csv_path),
        )

    # Keep only the six external ground-truth metrics used in Fig. S6.
    # This intentionally removes diagnostic/non-ground-truth metrics such as
    # external_mean_confidence.
    df = df[df[external_col].isin(EXTERNAL_METRIC_ORDER)].copy()

    # Remove internal confidence-only rows from both plot and output matrices.
    # mean_confidence is not treated as a scBOA objective component in this
    # ground-truth concordance heatmap.
    internal_norm = df[internal_col].astype(str).str.lower()
    df = df[~internal_norm.isin(INTERNAL_METRICS_EXCLUDE)].copy()

    if df.empty:
        raise ValueError(
            f"After filtering to expected external metrics, no rows remain in {csv_path}. "
            f"Observed external metric names: "
            f"{sorted(pd.read_csv(csv_path)[external_col].dropna().unique())}"
        )

    df[rho_col] = pd.to_numeric(df[rho_col], errors="coerce")

    mat = df.pivot_table(
        index=internal_col,
        columns=external_col,
        values=rho_col,
        aggfunc="mean",
    )

    return mat


def reorder_matrix(mat, external_order=EXTERNAL_METRIC_ORDER):
    """Reorder external metric columns and keep internal metrics in file order."""
    cols = [c for c in external_order if c in mat.columns]
    return mat.loc[:, cols]


def relabel_matrix(mat):
    """Apply human-readable labels to rows and columns."""
    mat = mat.copy()
    mat.index = [INTERNAL_LABELS.get(x, x) for x in mat.index]
    mat.columns = [EXTERNAL_LABELS.get(x, x) for x in mat.columns]
    return mat


def plot_heatmaps(raw_mat, partial_mat, output_prefix):
    """Plot raw and partial correlation heatmaps side by side."""

    all_rows = list(dict.fromkeys(list(raw_mat.index) + list(partial_mat.index)))
    all_cols = [
        c
        for c in EXTERNAL_METRIC_ORDER
        if c in raw_mat.columns or c in partial_mat.columns
    ]

    raw_mat = raw_mat.reindex(index=all_rows, columns=all_cols)
    partial_mat = partial_mat.reindex(index=all_rows, columns=all_cols)

    raw_plot = relabel_matrix(raw_mat)
    partial_plot = relabel_matrix(partial_mat)

    n_rows = max(raw_plot.shape[0], partial_plot.shape[0])
    n_cols = max(raw_plot.shape[1], partial_plot.shape[1])

    fig_width = max(12, 1.45 * n_cols * 2)
    fig_height = max(4.8, 0.6 * n_rows + 2.2)

    sns.set_theme(style="white", font_scale=0.95)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        raw_plot,
        ax=axes[0],
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        square=False,
    )

    sns.heatmap(
        partial_plot,
        ax=axes[1],
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Spearman rho"},
        square=False,
    )

    axes[0].set_title("Raw Spearman correlation", fontsize=13, weight="bold")
    axes[1].set_title(
        "Partial Spearman correlation\n"
        "controlling for n_leiden_clusters + resolution",
        fontsize=13,
        weight="bold",
    )

    for ax in axes:
        ax.set_xlabel("External PBMC ground-truth metric")
        ax.set_ylabel("scBOA internal metric")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "scBOA internal objective versus external PBMC ground-truth concordance",
        fontsize=15,
        weight="bold",
        y=1.04,
    )

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap PNG: {png_path}")
    print(f"Saved heatmap PDF: {pdf_path}")

    raw_matrix_path = output_prefix.parent / f"{output_prefix.name}_raw_matrix.csv"
    partial_matrix_path = output_prefix.parent / f"{output_prefix.name}_partial_matrix.csv"

    raw_mat.to_csv(raw_matrix_path)
    partial_mat.to_csv(partial_matrix_path)

    print(f"Saved raw matrix: {raw_matrix_path}")
    print(f"Saved partial matrix: {partial_matrix_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing spearman_internal_vs_facs.csv and "
            "spearman_internal_vs_facs_partial_controlling_cluster_count_and_resolution.csv."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="figures/pbmc_raw_partial_objective_heatmap",
        help="Output prefix for PNG/PDF heatmap.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    raw_csv = input_dir / "spearman_internal_vs_facs.csv"
    partial_csv = (
        input_dir
        / "spearman_internal_vs_facs_partial_controlling_cluster_count_and_resolution.csv"
    )

    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing raw correlation CSV: {raw_csv}")

    if not partial_csv.exists():
        raise FileNotFoundError(f"Missing partial correlation CSV: {partial_csv}")

    raw_mat = load_correlation_matrix(raw_csv, mode="raw")
    partial_mat = load_correlation_matrix(partial_csv, mode="partial")

    raw_mat = reorder_matrix(raw_mat)
    partial_mat = reorder_matrix(partial_mat)

    plot_heatmaps(raw_mat, partial_mat, args.output_prefix)


if __name__ == "__main__":
    main()