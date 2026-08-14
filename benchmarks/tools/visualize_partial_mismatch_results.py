#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

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


def read_input(h5ad_path, metadata_path, prediction_col, confidence_col):
    adata = sc.read_h5ad(h5ad_path)

    if prediction_col not in adata.obs.columns:
        raise ValueError(
            f"{prediction_col!r} not found in {h5ad_path}. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    adata.obs_names = adata.obs_names.astype(str)

    meta = pd.read_csv(metadata_path)
    required = {"cell_id", "ground_truth_cell_type"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    meta["cell_id"] = meta["cell_id"].astype(str)
    meta = meta.drop_duplicates("cell_id").set_index("cell_id")

    keep = adata.obs_names.intersection(meta.index)
    if len(keep) == 0:
        raise ValueError("No barcode overlap between h5ad and metadata.")

    obs = adata.obs.loc[keep].copy()
    obs["ground_truth_cell_type"] = meta.loc[
        keep, "ground_truth_cell_type"
    ].values
    obs[prediction_col] = obs[prediction_col].astype(str)

    if confidence_col in obs.columns:
        obs[confidence_col] = pd.to_numeric(
            obs[confidence_col], errors="coerce"
        )

    return obs


def hungarian_metrics(y_true, y_pred):
    true_labels = np.array(sorted(pd.unique(y_true)))
    pred_labels = np.array(sorted(pd.unique(y_pred)))

    cm = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    true_index = {x: i for i, x in enumerate(true_labels)}
    pred_index = {x: i for i, x in enumerate(pred_labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_index[true_label], pred_index[pred_label]] += 1

    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[pred_labels[c]] = true_labels[r]

    mapped = np.array([mapping.get(x, x) for x in y_pred])

    return (
        accuracy_score(y_true, mapped),
        f1_score(y_true, mapped, average="macro", zero_division=0),
        pd.DataFrame(cm, index=true_labels, columns=pred_labels),
    )


def calculate_metrics(df, label_col, prediction_col):
    sub = df[[label_col, prediction_col]].dropna().copy()
    sub[label_col] = sub[label_col].astype(str)
    sub[prediction_col] = sub[prediction_col].astype(str)

    y_true = sub[label_col].values
    y_pred = sub[prediction_col].values

    hung_acc, hung_f1, confusion = hungarian_metrics(y_true, y_pred)

    return {
        "n_cells": len(sub),
        "n_predicted_labels": sub[prediction_col].nunique(),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "FMI": fowlkes_mallows_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "V_measure": v_measure_score(y_true, y_pred),
        "hungarian_accuracy": hung_acc,
        "hungarian_macro_F1": hung_f1,
        "confusion": confusion,
    }


def save_confusion_plot(confusion, output_path, title):
    normalized = confusion.div(
        confusion.sum(axis=1).replace(0, np.nan), axis=0
    )

    plt.figure(figsize=(max(8, 0.55 * len(confusion.columns)),
                        max(5, 0.5 * len(confusion.index))))

    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Fraction within ground-truth class"},
    )

    plt.xlabel("Predicted label")
    plt.ylabel("Ground-truth label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metric_plot(summary, output_path):
    metric_cols = [
        "ARI",
        "NMI",
        "AMI",
        "FMI",
        "V_measure",
        "hungarian_accuracy",
        "hungarian_macro_F1",
    ]

    plot_df = summary.melt(
        id_vars="reference",
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(11, 5.5))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="value",
        hue="reference",
        palette={"full": "#3366CC", "noB": "#CC6633"},
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.xlabel("")
    plt.title("Full reference versus no-B reference")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_b_cells(
    full_df,
    nob_df,
    prediction_col,
    confidence_col,
    cas_col,
    output_dir,
):
    common = full_df.index.intersection(nob_df.index)

    comparison = pd.DataFrame(index=common)
    comparison["ground_truth"] = full_df.loc[
        common, "ground_truth_cell_type"
    ].astype(str)

    comparison["full_prediction"] = full_df.loc[
        common, prediction_col
    ].astype(str)

    comparison["noB_prediction"] = nob_df.loc[
        common, prediction_col
    ].astype(str)

    if confidence_col in full_df.columns:
        comparison["full_confidence"] = full_df.loc[
            common, confidence_col
        ]

    if confidence_col in nob_df.columns:
        comparison["noB_confidence"] = nob_df.loc[
            common, confidence_col
        ]

    if cas_col:
        if cas_col in full_df.columns:
            comparison["full_cas"] = full_df.loc[common, cas_col]
        if cas_col in nob_df.columns:
            comparison["noB_cas"] = nob_df.loc[common, cas_col]

    b = comparison[comparison["ground_truth"] == "B_cells"].copy()

    if b.empty:
        raise ValueError("No ground-truth B_cells found.")

    b["noB_is_error"] = b["noB_prediction"] != "B_cells"

    error_b = b[b["noB_is_error"]].copy()

    destination = (
        error_b["noB_prediction"]
        .value_counts()
        .rename_axis("noB_predicted_label")
        .reset_index(name="n_cells")
    )
    destination["fraction_of_B_cell_errors"] = (
        destination["n_cells"] / len(error_b)
    )
    destination["fraction_of_all_B_cells"] = (
        destination["n_cells"] / len(b)
    )
    destination.to_csv(
        output_dir / "B_cells_noB_error_destinations.csv",
        index=False,
    )

    b.to_csv(
        output_dir / "B_cells_full_vs_noB_per_cell.csv",
        index=True,
    )

    # Confidence summary
    confidence_rows = []
    for reference, column in [
        ("full", "full_confidence"),
        ("noB", "noB_confidence"),
    ]:
        if column not in b.columns:
            continue

        values_all = pd.to_numeric(b[column], errors="coerce").dropna()
        values_error = pd.to_numeric(
            error_b[column], errors="coerce"
        ).dropna()

        for subset_name, values in [
            ("all_B_cells", values_all),
            ("misclassified_B_cells", values_error),
        ]:
            if len(values) == 0:
                continue

            confidence_rows.append({
                "reference": reference,
                "subset": subset_name,
                "n_cells": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
                "q90": values.quantile(0.90),
                "fraction_ge_0.8": (values >= 0.8).mean(),
                "fraction_ge_0.9": (values >= 0.9).mean(),
            })

    confidence_summary = pd.DataFrame(confidence_rows)
    confidence_summary.to_csv(
        output_dir / "B_cells_confidence_summary.csv",
        index=False,
    )

    # Destination plot
    plt.figure(figsize=(max(7, 0.7 * len(destination)), 5))
    ax = sns.barplot(
        data=destination,
        x="noB_predicted_label",
        y="n_cells",
        color="#CC6633",
    )
    ax.set_xlabel("No-B predicted label")
    ax.set_ylabel("Number of ground-truth B cells")
    ax.set_title("Ground-truth B cells incorrectly assigned by no-B reference")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        output_dir / "B_cells_noB_error_destinations.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Confidence plot
    confidence_columns = [
        x for x in ["full_confidence", "noB_confidence"]
        if x in b.columns
    ]

    if confidence_columns:
        long_conf = b.melt(
            value_vars=confidence_columns,
            var_name="reference",
            value_name="confidence",
        )
        long_conf["reference"] = long_conf["reference"].map({
            "full_confidence": "full",
            "noB_confidence": "noB",
        })

        plt.figure(figsize=(7, 5))
        sns.violinplot(
            data=long_conf,
            x="reference",
            y="confidence",
            inner="quartile",
            cut=0,
            palette={"full": "#3366CC", "noB": "#CC6633"},
        )
        plt.ylim(0, 1)
        plt.ylabel("CellTypist confidence")
        plt.xlabel("")
        plt.title("Confidence for ground-truth B cells")
        plt.tight_layout()
        plt.savefig(
            output_dir / "B_cells_confidence_full_vs_noB.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # CAS status
    cas_columns = [
        x for x in ["full_cas", "noB_cas"] if x in b.columns
    ]

    with open(output_dir / "analysis_summary.txt", "w") as handle:
        handle.write("Partial label-space mismatch analysis\n")
        handle.write("=====================================\n\n")
        handle.write(f"Ground-truth B cells: {len(b)}\n")
        handle.write(f"Misclassified by no-B reference: {len(error_b)}\n")
        handle.write(
            f"Error fraction among B cells: "
            f"{len(error_b) / len(b):.4f}\n\n"
        )

        handle.write("Most common no-B error destinations:\n")
        handle.write(destination.head(10).to_string(index=False))
        handle.write("\n\n")

        if confidence_columns:
            handle.write(
                "Confidence columns were available and summarized in "
                "B_cells_confidence_summary.csv.\n"
            )
        else:
            handle.write("No confidence column was available.\n")

        if cas_columns:
            handle.write(
                "CAS columns were found and included in the per-cell table.\n"
            )
        else:
            handle.write(
                "No cell-level CAS column was found. CAS cannot be assessed "
                "from these h5ad files.\n"
            )

    return {
        "n_B_cells": len(b),
        "n_B_errors": len(error_b),
        "B_error_fraction": len(error_b) / len(b),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and summarize full versus no-B PBMC results."
    )
    parser.add_argument("--full_h5ad", required=True)
    parser.add_argument("--nob_h5ad", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument(
        "--prediction_col",
        default="ctpt_consensus_prediction",
    )
    parser.add_argument(
        "--confidence_col",
        default="ctpt_confidence",
    )
    parser.add_argument(
        "--cas_col",
        default="",
        help="Optional cell-level CAS column, e.g. cell_cas.",
    )
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full = read_input(
        args.full_h5ad,
        args.metadata_csv,
        args.prediction_col,
        args.confidence_col,
    )
    nob = read_input(
        args.nob_h5ad,
        args.metadata_csv,
        args.prediction_col,
        args.confidence_col,
    )

    common = full.index.intersection(nob.index)
    if len(common) == 0:
        raise ValueError("Full and no-B h5ad files have no common barcodes.")

    full = full.loc[common]
    nob = nob.loc[common]

    metrics = []
    confusion_tables = {}

    for reference, data in [("full", full), ("noB", nob)]:
        result = calculate_metrics(
            data,
            "ground_truth_cell_type",
            args.prediction_col,
        )
        confusion_tables[reference] = result.pop("confusion")
        result["reference"] = reference
        metrics.append(result)

    summary = pd.DataFrame(metrics)
    summary = summary[
        ["reference"] + [
            c for c in summary.columns if c != "reference"
        ]
    ]
    summary.to_csv(output_dir / "full_vs_noB_metrics.csv", index=False)

    save_metric_plot(
        summary,
        output_dir / "full_vs_noB_metric_comparison.png",
    )

    for reference, confusion in confusion_tables.items():
        confusion.to_csv(
            output_dir / f"{reference}_confusion_matrix_counts.csv"
        )
        save_confusion_plot(
            confusion,
            output_dir / f"{reference}_normalized_confusion_matrix.png",
            f"{reference} reference: normalized confusion matrix",
        )

    b_summary = analyze_b_cells(
        full,
        nob,
        args.prediction_col,
        args.confidence_col,
        args.cas_col if args.cas_col else None,
        output_dir,
    )

    print("\nAnalysis completed")
    print("==================")
    print(f"Common cells: {len(common)}")
    print(summary.to_string(index=False))
    print("\nB-cell summary:")
    print(f"Ground-truth B cells: {b_summary['n_B_cells']}")
    print(f"No-B errors: {b_summary['n_B_errors']}")
    print(f"No-B error fraction: {b_summary['B_error_fraction']:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()