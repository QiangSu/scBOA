#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark scBOA against reviewer-listed parameter-selection / clustering
methods on datasets with external ground-truth labels.

Reviewer Comment 4 (Remaining Concerns) explicitly lists:
    Sciaraffa et al. (Front. Bioinform. 2025, 5:1562410)
    Fang et al. (Genome Biol. 2024, 25:127)
    scclusteval (Bioinformatics 2021)
    chooseR (BMC Bioinformatics 2021)
    MultiK (Genome Biology 2021)
    scICE (Nat. Commun. 2025)
    scLENS (Nat. Commun. 2024)
    findPC (Bioinformatics 2022)

Additional principled baselines run alongside the reviewer-listed set:
    clustree           (Zappia & Oshlack, GigaScience 2018)
    silhouette-max     (max mean silhouette across a resolution sweep)
    mclust_BIC         (Gaussian mixture BIC selection on PCA embedding)
    eigengap           (spectral eigengap heuristic for n_pcs)

Each method produces a clustering / annotation evaluated against the same
external FACS / expert labels used elsewhere in the rebuttal. This script
consolidates per-method summaries and produces:

    - all_methods_all_prediction_columns.csv
    - comparison_<prediction_col>.csv
    - method_ranking_by_metric_<prediction_col>.csv
    - delta_vs_scboa_<prediction_col>.csv
    - barplot_<prediction_col>_<metric>.{png,pdf}
    - grouped_barplot_<prediction_col>.{png,pdf}

Note: AMI and V_measure are numerically near-identical to NMI on these
datasets; only NMI is retained in outputs to avoid visual redundancy.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
# Metric definitions
# ------------------------------------------------------------------ #

# NMI kept; AMI and V_measure dropped from outputs/plots (redundant with NMI).
KNOWN_METRIC_COLS = [
    "ARI", "NMI", "FMI",
    "homogeneity", "completeness",
    "direct_accuracy", "direct_macro_F1",
    "hungarian_accuracy", "hungarian_macro_F1",
    "n_cells", "n_cells_evaluated",
    "n_facs_labels", "n_predicted_labels",
]

RANK_METRICS_DEFAULT = [
    "ARI", "NMI",
    "homogeneity", "completeness",
    "hungarian_accuracy", "hungarian_macro_F1",
    "direct_accuracy", "direct_macro_F1",
]

# Columns to drop from the combined table if they appear in input CSVs.
DROPPED_REDUNDANT_METRICS = ["AMI", "V_measure"]


# ------------------------------------------------------------------ #
# I/O helpers
# ------------------------------------------------------------------ #

def parse_method_arg(method_arg):
    if "=" not in method_arg:
        raise ValueError(
            f"Invalid --method format: {method_arg}\n"
            "Expected: MethodName=/path/to/summary.csv"
        )
    name, path = method_arg.split("=", 1)
    name = name.strip()
    path = Path(path.strip())
    if not name:
        raise ValueError(f"Empty method name in --method: {method_arg}")
    return name, path


def add_method_if_provided(method_files, method_name, csv_path):
    if csv_path is not None:
        method_files[method_name] = Path(csv_path)


def standardize_numeric_columns(df):
    for col in KNOWN_METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Make n_cells and n_cells_evaluated interchangeable for downstream display
    if "n_cells_evaluated" in df.columns and "n_cells" not in df.columns:
        df["n_cells"] = df["n_cells_evaluated"]
    # Drop the redundant metrics if present in the input files.
    to_drop = [c for c in DROPPED_REDUNDANT_METRICS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def load_method_summary(method_name, csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARNING] Missing file for {method_name}: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if "method" in df.columns:
        df = df.drop(columns=["method"])
    if "source_file" in df.columns:
        df = df.drop(columns=["source_file"])
    df.insert(0, "method", method_name)
    df.insert(1, "source_file", str(csv_path))
    df = standardize_numeric_columns(df)
    return df


# ------------------------------------------------------------------ #
# Analysis
# ------------------------------------------------------------------ #

def build_ranking_table(combined, prediction_col, scboa_name, rank_metrics):
    sub = combined[combined["prediction_column"] == prediction_col].copy()
    if sub.empty:
        return None

    metrics_present = [m for m in rank_metrics if m in sub.columns]
    ranking = sub[["method"] + metrics_present].copy()

    for m in metrics_present:
        ranking[f"rank_{m}"] = ranking[m].rank(ascending=False, method="min",
                                               na_option="bottom")

    rank_cols = [f"rank_{m}" for m in metrics_present]
    ranking["mean_rank"] = ranking[rank_cols].mean(axis=1)
    ranking = ranking.sort_values("mean_rank")
    ranking["is_scBOA"] = (ranking["method"] == scboa_name)
    return ranking


def build_delta_vs_scboa(combined, prediction_col, scboa_name, rank_metrics):
    sub = combined[combined["prediction_column"] == prediction_col].copy()
    if sub.empty:
        return None
    if scboa_name not in sub["method"].values:
        print(f"[WARNING] scBOA method '{scboa_name}' not found for delta table.")
        return None

    scboa_row = sub[sub["method"] == scboa_name].iloc[0]
    metrics_present = [m for m in rank_metrics if m in sub.columns]

    rows = []
    for _, r in sub.iterrows():
        rec = {"method": r["method"], "prediction_column": prediction_col}
        for m in metrics_present:
            v = r[m]
            v_ref = scboa_row[m]
            rec[m] = v
            rec[f"delta_{m}_vs_scBOA"] = (
                (v - v_ref) if pd.notna(v) and pd.notna(v_ref) else np.nan
            )
            if pd.notna(v) and pd.notna(v_ref) and v_ref != 0:
                rec[f"pct_{m}_vs_scBOA"] = 100.0 * (v - v_ref) / abs(v_ref)
            else:
                rec[f"pct_{m}_vs_scBOA"] = np.nan
        rows.append(rec)

    delta = pd.DataFrame(rows)
    primary = metrics_present[0] if metrics_present else None
    if primary is not None:
        delta["_is_scboa"] = delta["method"] == scboa_name
        delta = delta.sort_values(
            ["_is_scboa", f"delta_{primary}_vs_scBOA"],
            ascending=[False, True],
        ).drop(columns=["_is_scboa"])
    return delta


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

def _try_import_plotting():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError:
        print("[WARNING] matplotlib/seaborn not installed. Skipping plots.")
        return None, None


def _filter_redundant(metrics):
    """Silently drop AMI / V_measure if callers still request them."""
    return [m for m in metrics if m not in DROPPED_REDUNDANT_METRICS]


def make_barplots(df, outdir, prediction_col, metrics, scboa_name):
    plt, sns = _try_import_plotting()
    if plt is None:
        return

    metrics = _filter_redundant(metrics)

    plot_df = df[df["prediction_column"] == prediction_col].copy()
    if plot_df.empty:
        print(f"[WARNING] No rows for prediction column: {prediction_col}")
        return

    for metric in metrics:
        if metric not in plot_df.columns:
            print(f"[WARNING] Metric not found, skipping: {metric}")
            continue

        p = plot_df.dropna(subset=[metric]).copy()
        if p.empty:
            continue

        p = p.sort_values(metric, ascending=False)

        colors = ["#C0392B" if m == scboa_name else "#4C72B0"
                  for m in p["method"]]

        width = max(7, 0.8 * len(p))
        plt.figure(figsize=(width, 4.8))
        ax = plt.gca()
        ax.bar(p["method"], p[metric], color=colors)

        if scboa_name in p["method"].values:
            ref = p.loc[p["method"] == scboa_name, metric].iloc[0]
            ax.axhline(ref, color="#C0392B", linestyle="--", linewidth=1,
                       alpha=0.6, label=f"{scboa_name} = {ref:.3f}")
            ax.legend(loc="best", fontsize=9)

        plt.xticks(rotation=35, ha="right")
        plt.xlabel("")
        plt.ylabel(metric)
        plt.title(f"{metric} by method\nPrediction: {prediction_col}")

        for i, v in enumerate(p[metric].values):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        clean_pred = prediction_col.replace("/", "_").replace(" ", "_")
        for ext in ("png",):
            out = outdir / f"barplot_{clean_pred}_{metric}.{ext}"
            plt.savefig(out, dpi=300 if ext == "png" else None)
        plt.close()
        print(f"Saved per-metric barplot: barplot_{clean_pred}_{metric}.png")


def make_grouped_barplot(df, outdir, prediction_col, metrics, scboa_name):
    plt, sns = _try_import_plotting()
    if plt is None:
        return

    metrics = _filter_redundant(metrics)

    plot_df = df[df["prediction_column"] == prediction_col].copy()
    if plot_df.empty:
        return

    metrics_present = [m for m in metrics if m in plot_df.columns]
    if not metrics_present:
        return

    long_df = plot_df.melt(
        id_vars=["method"],
        value_vars=metrics_present,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    order = (long_df.groupby("method")["value"].mean()
             .sort_values(ascending=False).index.tolist())
    long_df["method"] = pd.Categorical(long_df["method"], categories=order, ordered=True)

    width = max(9, 1.1 * len(order))
    plt.figure(figsize=(width, 5.2))
    ax = plt.gca()

    sns.barplot(
        data=long_df, x="method", y="value", hue="metric",
        ax=ax, edgecolor="white",
    )

    if scboa_name in order:
        xi = order.index(scboa_name)
        ax.axvspan(xi - 0.5, xi + 0.5, color="#C0392B", alpha=0.08, zorder=0)

    plt.xticks(rotation=35, ha="right")
    plt.xlabel("")
    plt.ylabel("Metric value")
    plt.title(f"External-label metrics across methods\nPrediction: {prediction_col}")
    plt.legend(title="metric", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    clean_pred = prediction_col.replace("/", "_").replace(" ", "_")
    for ext in ("png", ):
        out = outdir / f"grouped_barplot_{clean_pred}.{ext}"
        plt.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved grouped barplot: grouped_barplot_{clean_pred}.png")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark scBOA against reviewer-listed parameter-selection / "
            "clustering methods on datasets with external ground-truth labels."
        )
    )

    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prediction_col", default="ctpt_consensus_prediction")
    parser.add_argument(
        "--extra_prediction_cols", nargs="*",
        default=["leiden", "ctpt_individual_prediction"],
    )
    parser.add_argument("--sort_metric", default="ARI")

    # Optimizer-comparison group
    parser.add_argument("--default_csv", default=None)
    parser.add_argument("--random_csv", default=None)
    parser.add_argument("--optuna_csv", default=None)
    parser.add_argument("--scboa_csv", default=None)
    parser.add_argument("--default_name", default="Default")
    parser.add_argument("--random_name", default="Random search")
    parser.add_argument("--optuna_name", default="Optuna-TPE")
    parser.add_argument("--scboa_name", default="scBOA")

    # Reviewer-listed baselines
    parser.add_argument("--clustree_csv", default=None)
    parser.add_argument("--chooseR_csv", default=None)
    parser.add_argument("--multik_csv", default=None,
                        help="MultiK / multik_style resolution selection summary.")
    parser.add_argument("--scclusteval_csv", default=None)
    parser.add_argument("--scice_csv", default=None)
    parser.add_argument("--sclens_csv", default=None)
    parser.add_argument("--findpc_csv", default=None)
    parser.add_argument("--sciaraffa_csv", default=None)
    parser.add_argument("--fang_csv", default=None)

    # Additional principled baselines
    parser.add_argument("--silhouette_max_csv", default=None,
                        help="Maximum-mean-silhouette resolution selection summary.")
    parser.add_argument("--mclust_bic_csv", default=None,
                        help="mclust BIC-based cluster count selection summary.")
    parser.add_argument("--eigengap_csv", default=None,
                        help="Spectral eigengap n_pcs selection summary.")

    # Generic escape hatch
    parser.add_argument(
        "--method", action="append", default=[],
        help="MethodName=/path/to/summary.csv (repeatable).",
    )

    # Plot config
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument(
        "--plot_metrics", nargs="*",
        default=[
            "ARI", "NMI",
            "homogeneity", "completeness",
            "hungarian_accuracy", "hungarian_macro_F1",
        ],
    )
    parser.add_argument(
        "--rank_metrics", nargs="*",
        default=RANK_METRICS_DEFAULT,
    )

    args = parser.parse_args()

    # Enforce removal of redundant metrics even if user passes them explicitly.
    args.plot_metrics = _filter_redundant(args.plot_metrics)
    args.rank_metrics = _filter_redundant(args.rank_metrics)

    base_dir = Path(args.base_dir)
    output_dir = (Path(args.output_dir) if args.output_dir
                  else base_dir / "method_comparison_reviewer")
    output_dir.mkdir(parents=True, exist_ok=True)

    method_files = {}
    # Optimizer comparison group
    add_method_if_provided(method_files, args.default_name, args.default_csv)
    add_method_if_provided(method_files, args.random_name,  args.random_csv)
    add_method_if_provided(method_files, args.optuna_name,  args.optuna_csv)
    add_method_if_provided(method_files, args.scboa_name,   args.scboa_csv)

    # Reviewer-listed baselines
    add_method_if_provided(method_files, "clustree",       args.clustree_csv)
    add_method_if_provided(method_files, "chooseR",        args.chooseR_csv)
    add_method_if_provided(method_files, "MultiK",         args.multik_csv)
    add_method_if_provided(method_files, "scclusteval",    args.scclusteval_csv)
    add_method_if_provided(method_files, "scICE",          args.scice_csv)
    add_method_if_provided(method_files, "scLENS",         args.sclens_csv)
    add_method_if_provided(method_files, "findPC",         args.findpc_csv)
    add_method_if_provided(method_files, "Sciaraffa2025",  args.sciaraffa_csv)
    add_method_if_provided(method_files, "Fang2024",       args.fang_csv)

    # Additional principled baselines
    add_method_if_provided(method_files, "silhouette_max", args.silhouette_max_csv)
    add_method_if_provided(method_files, "mclust_BIC",     args.mclust_bic_csv)
    add_method_if_provided(method_files, "eigengap",       args.eigengap_csv)

    for m in args.method:
        name, path = parse_method_arg(m)
        method_files[name] = path

    if not method_files:
        raise SystemExit("No methods provided.")

    print("=" * 60)
    print("Reviewer-comment-4 benchmark: scBOA vs listed methods")
    print("=" * 60)
    print(f"Base directory:   {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Prediction col:   {args.prediction_col}")
    print(f"Sort metric:      {args.sort_metric}")
    print(f"scBOA method key: {args.scboa_name}")
    print(f"Dropped metrics:  {DROPPED_REDUNDANT_METRICS} "
          f"(AMI and V_measure are redundant with NMI on these datasets)")
    print("Methods:")
    for method, path in method_files.items():
        print(f"  - {method}: {path}")
    print("=" * 60)

    all_rows = []
    for method_name, csv_path in method_files.items():
        df = load_method_summary(method_name, csv_path)
        if df is not None:
            all_rows.append(df)
    if not all_rows:
        raise SystemExit("No valid summary CSV files were found.")

    combined = pd.concat(all_rows, ignore_index=True)

    # Safety net: drop any redundant metrics that survived after concat.
    to_drop_after = [c for c in DROPPED_REDUNDANT_METRICS if c in combined.columns]
    if to_drop_after:
        combined = combined.drop(columns=to_drop_after)

    preferred_cols = [
        "method", "source_file", "prediction_column", "facs_column",
        "n_cells", "n_cells_evaluated", "n_facs_labels", "n_predicted_labels",
        "ARI", "NMI", "FMI",
        "homogeneity", "completeness",
        "direct_accuracy", "direct_macro_F1",
        "hungarian_accuracy", "hungarian_macro_F1",
    ]
    existing_preferred_cols = [c for c in preferred_cols if c in combined.columns]
    remaining_cols = [c for c in combined.columns if c not in existing_preferred_cols]
    combined = combined[existing_preferred_cols + remaining_cols]

    all_out = output_dir / "all_methods_all_prediction_columns.csv"
    combined.to_csv(all_out, index=False)
    print(f"\nSaved full combined table: {all_out}")

    prediction_cols_to_export = [args.prediction_col] + args.extra_prediction_cols

    for pred_col in prediction_cols_to_export:
        sub = combined[combined["prediction_column"] == pred_col].copy()
        if sub.empty:
            print(f"\n[WARNING] No rows for prediction column: {pred_col}")
            continue
        if args.sort_metric in sub.columns:
            sub = sub.sort_values(args.sort_metric, ascending=False)

        clean_pred = pred_col.replace("/", "_").replace(" ", "_")
        out = output_dir / f"comparison_{clean_pred}.csv"
        sub.to_csv(out, index=False)

        print("\n" + "=" * 60)
        print(f"Comparison for prediction column: {pred_col}")
        print("=" * 60)
        display_cols = [
            "method", "prediction_column",
            "n_cells", "n_cells_evaluated", "n_facs_labels", "n_predicted_labels",
            "ARI", "NMI",
            "homogeneity", "completeness",
            "hungarian_accuracy", "hungarian_macro_F1",
        ]
        display_cols = [c for c in display_cols if c in sub.columns]
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(sub[display_cols].to_string(index=False))
        print(f"Saved: {out}")

        ranking = build_ranking_table(combined, pred_col,
                                      args.scboa_name, args.rank_metrics)
        if ranking is not None:
            rank_out = output_dir / f"method_ranking_by_metric_{clean_pred}.csv"
            ranking.to_csv(rank_out, index=False)
            print(f"Saved ranking table: {rank_out}")
            print("\nMean-rank across metrics (lower is better):")
            print(ranking[["method", "mean_rank", "is_scBOA"]].to_string(index=False))

        delta = build_delta_vs_scboa(combined, pred_col,
                                     args.scboa_name, args.rank_metrics)
        if delta is not None:
            d_out = output_dir / f"delta_vs_scboa_{clean_pred}.csv"
            delta.to_csv(d_out, index=False)
            print(f"Saved delta-vs-scBOA table: {d_out}")

    if args.make_plots:
        for pred_col in prediction_cols_to_export:
            make_barplots(combined, output_dir, pred_col,
                          args.plot_metrics, args.scboa_name)
            make_grouped_barplot(combined, output_dir, pred_col,
                                 args.plot_metrics, args.scboa_name)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()