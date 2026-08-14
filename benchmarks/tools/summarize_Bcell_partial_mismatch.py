#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # 第一列通常为 cell barcode
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.rename(columns={unnamed[0]: "cell_id"})

    required = [
        "ground_truth",
        "full_prediction",
        "noB_prediction",
        "full_confidence",
        "noB_confidence",
        "noB_is_error",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df["full_confidence"] = pd.to_numeric(
        df["full_confidence"], errors="coerce"
    )
    df["noB_confidence"] = pd.to_numeric(
        df["noB_confidence"], errors="coerce"
    )

    df["noB_is_error"] = (
        df["noB_is_error"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(df["noB_is_error"])
        .astype(bool)
    )

    df = df.dropna(subset=["full_confidence", "noB_confidence"]).copy()

    # ---------------------------------------------------------
    # 1. 总体 confidence summary
    # ---------------------------------------------------------
    summary_rows = []

    for reference, col in [
        ("full", "full_confidence"),
        ("noB", "noB_confidence"),
    ]:
        x = df[col].to_numpy()

        summary_rows.append({
            "reference": reference,
            "n_cells": len(x),
            "mean_confidence": np.mean(x),
            "sd_confidence": np.std(x, ddof=1),
            "median_confidence": np.median(x),
            "q25_confidence": np.quantile(x, 0.25),
            "q75_confidence": np.quantile(x, 0.75),
            "q90_confidence": np.quantile(x, 0.90),
            "q95_confidence": np.quantile(x, 0.95),
            "min_confidence": np.min(x),
            "max_confidence": np.max(x),
        })

    confidence_summary = pd.DataFrame(summary_rows)
    confidence_summary.to_csv(
        outdir / "B_cell_confidence_summary.csv", index=False
    )

    # ---------------------------------------------------------
    # 2. Paired difference
    # ---------------------------------------------------------
    df["confidence_change_noB_minus_full"] = (
        df["noB_confidence"] - df["full_confidence"]
    )

    try:
        stat, pvalue = wilcoxon(
            df["full_confidence"],
            df["noB_confidence"],
            alternative="greater",
        )
    except ValueError:
        stat, pvalue = np.nan, np.nan

    paired_summary = pd.DataFrame([{
        "n_cells": len(df),
        "full_mean": df["full_confidence"].mean(),
        "noB_mean": df["noB_confidence"].mean(),
        "mean_change_noB_minus_full":
            df["confidence_change_noB_minus_full"].mean(),
        "full_median": df["full_confidence"].median(),
        "noB_median": df["noB_confidence"].median(),
        "median_change_noB_minus_full":
            df["confidence_change_noB_minus_full"].median(),
        "wilcoxon_alternative": "full > noB",
        "wilcoxon_statistic": stat,
        "wilcoxon_pvalue": pvalue,
    }])

    paired_summary.to_csv(
        outdir / "B_cell_paired_confidence_test.csv", index=False
    )

    # ---------------------------------------------------------
    # 3. 错误标签 destination
    # ---------------------------------------------------------
    errors = df[df["noB_is_error"]].copy()

    destination = (
        errors.groupby("noB_prediction", observed=True)
        .agg(
            n_cells=("noB_prediction", "size"),
            mean_noB_confidence=("noB_confidence", "mean"),
            median_noB_confidence=("noB_confidence", "median"),
            q25_noB_confidence=("noB_confidence",
                                lambda x: x.quantile(0.25)),
            q75_noB_confidence=("noB_confidence",
                                lambda x: x.quantile(0.75)),
        )
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )

    destination["fraction_of_B_cell_errors"] = (
        destination["n_cells"] / len(errors)
    )
    destination["fraction_of_all_B_cells"] = (
        destination["n_cells"] / len(df)
    )

    destination.to_csv(
        outdir / "B_cell_error_destination_confidence.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Silent-error rates at several confidence thresholds
    # ---------------------------------------------------------
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90]

    threshold_rows = []

    for threshold in thresholds:
        n_high_conf_error = int(
            (
                df["noB_is_error"]
                & (df["noB_confidence"] >= threshold)
            ).sum()
        )

        threshold_rows.append({
            "confidence_threshold": threshold,
            "n_B_cells": len(df),
            "n_noB_errors": int(df["noB_is_error"].sum()),
            "n_errors_above_threshold": n_high_conf_error,
            "fraction_of_errors_above_threshold":
                n_high_conf_error / max(df["noB_is_error"].sum(), 1),
            "fraction_of_all_B_cells_above_threshold":
                n_high_conf_error / len(df),
        })

    threshold_summary = pd.DataFrame(threshold_rows)
    threshold_summary.to_csv(
        outdir / "B_cell_high_confidence_error_rates.csv",
        index=False,
    )

    # 使用 full-reference B-cell confidence 的分位数作为相对阈值
    relative_thresholds = {
        "full_B_q25": df["full_confidence"].quantile(0.25),
        "full_B_median": df["full_confidence"].median(),
        "full_B_q75": df["full_confidence"].quantile(0.75),
    }

    relative_rows = []

    for threshold_name, threshold in relative_thresholds.items():
        mask = (
            df["noB_is_error"]
            & (df["noB_confidence"] >= threshold)
        )

        relative_rows.append({
            "threshold_name": threshold_name,
            "confidence_threshold": threshold,
            "n_errors_above_threshold": int(mask.sum()),
            "fraction_of_errors_above_threshold":
                mask.sum() / max(df["noB_is_error"].sum(), 1),
        })

    pd.DataFrame(relative_rows).to_csv(
        outdir / "B_cell_relative_high_confidence_error_rates.csv",
        index=False,
    )

    # 保存带变化值的 per-cell 文件
    df.to_csv(
        outdir / "B_cells_full_vs_noB_with_confidence_change.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Plotting
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid", context="talk")

    # A. Destination labels
    plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        data=destination,
        x="noB_prediction",
        y="fraction_of_all_B_cells",
        color="#D55E00",
    )

    ax.set_xlabel("Predicted label under no-B reference")
    ax.set_ylabel("Fraction of ground-truth B cells")
    ax.set_title("Destinations of misassigned B cells")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(
        outdir / "B_cell_error_destinations.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        outdir / "B_cell_error_destinations.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # B. Full vs no-B confidence
    long_df = df.melt(
        id_vars=["ground_truth"],
        value_vars=["full_confidence", "noB_confidence"],
        var_name="reference",
        value_name="confidence",
    )

    long_df["reference"] = long_df["reference"].replace({
        "full_confidence": "Full reference",
        "noB_confidence": "No-B reference",
    })

    plt.figure(figsize=(7, 6))

    ax = sns.violinplot(
        data=long_df,
        x="reference",
        y="confidence",
        hue="reference",
        palette={
            "Full reference": "#0072B2",
            "No-B reference": "#D55E00",
        },
        legend=False,
        inner="box",
        cut=0,
    )

    ax.set_xlabel("")
    ax.set_ylabel("CellTypist confidence")
    ax.set_title("Confidence for ground-truth B cells")
    plt.tight_layout()
    plt.savefig(
        outdir / "B_cell_confidence_full_vs_noB.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        outdir / "B_cell_confidence_full_vs_noB.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # C. ECDF
    plt.figure(figsize=(8, 6))

    sns.ecdfplot(
        data=long_df,
        x="confidence",
        hue="reference",
        palette={
            "Full reference": "#0072B2",
            "No-B reference": "#D55E00",
        },
        linewidth=2.5,
    )

    plt.xlabel("CellTypist confidence")
    plt.ylabel("Cumulative fraction of B cells")
    plt.title("Confidence distributions under full and no-B references")
    plt.tight_layout()
    plt.savefig(
        outdir / "B_cell_confidence_ecdf.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        outdir / "B_cell_confidence_ecdf.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # D. confidence threshold plot
    plt.figure(figsize=(8, 5))

    ax = sns.lineplot(
        data=threshold_summary,
        x="confidence_threshold",
        y="fraction_of_errors_above_threshold",
        marker="o",
        color="#D55E00",
        linewidth=2.5,
    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Fraction of erroneous B-cell predictions")
    ax.set_title("High-confidence error rate under no-B reference")
    plt.tight_layout()
    plt.savefig(
        outdir / "B_cell_high_confidence_error_rate.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        outdir / "B_cell_high_confidence_error_rate.pdf",
        bbox_inches="tight",
    )
    plt.close()

    print("\nAnalysis completed")
    print("==================")
    print(confidence_summary.to_string(index=False))
    print("\nPaired confidence test:")
    print(paired_summary.to_string(index=False))
    print("\nError destinations:")
    print(destination.to_string(index=False))
    print("\nThreshold summary:")
    print(threshold_summary.to_string(index=False))
    print(f"\nResults saved to: {outdir}")


if __name__ == "__main__":
    main()