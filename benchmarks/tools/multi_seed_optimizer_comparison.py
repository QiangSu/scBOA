#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-seed optimizer comparison for scBOA rebuttal.

For each optimizer (scBOA_GP, Random_search, Optuna_TPE) the user provides one
stage_1_bayesian_optimization directory per seed. Within each seed's directory,
the *_yield_scores_report.csv is treated as one independent run. If multiple
strategies exist in that file, they are concatenated in (strategy, call_number)
order and the best-so-far curve is computed across the concatenated trials, so
each (optimizer, seed) contributes exactly one convergence trace.

Outputs (all written to --output_dir):
    per_seed_best_score.csv           best objective per (optimizer, seed)
    per_seed_best_params.csv          argmax parameter vector per (optimizer, seed)
    optimizer_best_score_summary.csv  mean/std/min/max of best score over seeds
    optimizer_param_summary.csv       mean/std of each optimized parameter over seeds
    convergence_mean_ci.csv           per (optimizer, eval_index) mean and 95% CI
    pairwise_wilcoxon_best_score.csv  pairwise Wilcoxon signed-rank tests
    convergence_curve_mean_ci.png     mean best-so-far with 95% CI band
    best_score_boxplot.png            per-optimizer best-score distribution over seeds
                                      with pairwise Wilcoxon p-values annotated
                                      and per-point seed labels
    param_variance_barplot.png        coefficient of variation of each parameter
"""

import os
import glob
import argparse
import warnings
import itertools

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# File discovery and loading
# -----------------------------------------------------------------------------
def find_yield_report(stage_dir):
    pattern = os.path.join(stage_dir, "*_yield_scores_report.csv")
    files = [f for f in glob.glob(pattern) if "refinement_depth" not in f]

    if len(files) == 0:
        raise FileNotFoundError(f"No *_yield_scores_report.csv found in: {stage_dir}")
    if len(files) > 1:
        print(f"[Warning] Multiple yield reports in {stage_dir}, using: {files[0]}")
    return files[0]


def infer_objective_column(df):
    candidates = [
        "yield_score_target", "balanced_score_gmean",
        "objective", "objective_score", "score", "target",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Cannot infer objective column.")


PARAM_COLS = [
    "n_hvg", "n_pcs", "n_neighbors", "resolution",
    "mean_cas", "weighted_mean_cas", "simple_mean_cas",
    "mean_mcs", "mean_f1", "mean_confidence", "silhouette_score",
    "n_individual_labels", "n_consensus_labels",
    "Final_n_individual_labels", "Final_n_consensus_labels",
]


def load_run(stage_dir, optimizer_name, seed, objective_col, max_trials):
    report = find_yield_report(stage_dir)
    df = pd.read_csv(report)

    if objective_col not in df.columns:
        raise ValueError(f"'{objective_col}' missing in {report}")

    if "strategy" not in df.columns or df["strategy"].isna().all():
        df["strategy"] = optimizer_name
    df["strategy"] = df["strategy"].fillna(optimizer_name).astype(str)

    if "call_number" not in df.columns:
        df["call_number"] = np.arange(1, len(df) + 1)

    df = df.sort_values(["strategy", "call_number"]).reset_index(drop=True)
    df["objective"] = pd.to_numeric(df[objective_col], errors="coerce")
    df = df.dropna(subset=["objective"]).reset_index(drop=True)

    if max_trials is not None:
        df = df.head(max_trials).copy()

    df["eval_index"] = np.arange(1, len(df) + 1)
    df["optimizer"] = optimizer_name
    df["seed"] = seed
    df["source_report"] = report
    df["best_so_far"] = df["objective"].cummax()

    return df


# -----------------------------------------------------------------------------
# Per-seed summaries
# -----------------------------------------------------------------------------
def summarize_run(run_df):
    best_idx = run_df["objective"].idxmax()
    best_row = run_df.loc[best_idx]

    rec = {
        "optimizer":         run_df["optimizer"].iloc[0],
        "seed":              run_df["seed"].iloc[0],
        "n_trials":          len(run_df),
        "best_score":        float(best_row["objective"]),
        "best_eval_index":   int(best_row["eval_index"]),
        "best_call_number":  best_row.get("call_number", np.nan),
        "best_strategy":     best_row.get("strategy", np.nan),
        "objective_mean":    float(run_df["objective"].mean()),
        "objective_median":  float(run_df["objective"].median()),
        "objective_sd":      float(run_df["objective"].std()),
        "auc_best_so_far":   float(np.trapz(run_df["best_so_far"].values, dx=1)
                                   / max(len(run_df) - 1, 1)),
        "source_report":     run_df["source_report"].iloc[0],
    }
    return rec, best_row


def extract_best_params(best_row):
    out = {}
    for c in PARAM_COLS:
        if c in best_row.index:
            out[c] = best_row[c]
    return out


# -----------------------------------------------------------------------------
# Aggregation across seeds
# -----------------------------------------------------------------------------
def aggregate_best_score(per_seed_df):
    rows = []
    for opt, sub in per_seed_df.groupby("optimizer"):
        vals = sub["best_score"].values
        rows.append({
            "optimizer":       opt,
            "n_seeds":         len(vals),
            "best_score_mean": float(np.mean(vals)),
            "best_score_std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "best_score_min":  float(np.min(vals)),
            "best_score_max":  float(np.max(vals)),
            "best_score_cv":   float(np.std(vals, ddof=1) / np.mean(vals))
                                if len(vals) > 1 and np.mean(vals) != 0 else np.nan,
            "seeds":           ",".join(map(str, sorted(sub["seed"].tolist()))),
        })
    return pd.DataFrame(rows)


def aggregate_params(per_seed_params_df):
    rows = []
    for opt, sub in per_seed_params_df.groupby("optimizer"):
        for c in PARAM_COLS:
            if c not in sub.columns:
                continue
            vals = pd.to_numeric(sub[c], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            cv   = float(std / mean) if (len(vals) > 1 and mean != 0) else np.nan
            rows.append({
                "optimizer":  opt,
                "parameter":  c,
                "n_seeds":    len(vals),
                "mean":       mean,
                "std":        std,
                "min":        float(np.min(vals)),
                "max":        float(np.max(vals)),
                "cv":         cv,
                "values":     ",".join(f"{v:g}" for v in vals),
            })
    return pd.DataFrame(rows)


def aggregate_convergence(all_runs_df):
    rows = []
    for (opt, ev), sub in all_runs_df.groupby(["optimizer", "eval_index"]):
        vals = sub["best_so_far"].values
        n = len(vals)
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        if n > 1:
            tcrit = stats.t.ppf(0.975, df=n - 1)
            half = tcrit * s / np.sqrt(n)
        else:
            half = 0.0
        rows.append({
            "optimizer":  opt,
            "eval_index": int(ev),
            "n_seeds":    n,
            "mean":       m,
            "std":        s,
            "ci95_low":   m - half,
            "ci95_high":  m + half,
        })
    return pd.DataFrame(rows).sort_values(["optimizer", "eval_index"]).reset_index(drop=True)


def pairwise_wilcoxon(per_seed_df):
    optimizers = sorted(per_seed_df["optimizer"].unique())
    rows = []

    for a, b in itertools.combinations(optimizers, 2):
        sub_a = per_seed_df[per_seed_df["optimizer"] == a][["seed", "best_score"]]
        sub_b = per_seed_df[per_seed_df["optimizer"] == b][["seed", "best_score"]]
        merged = sub_a.merge(sub_b, on="seed", suffixes=(f"_{a}", f"_{b}"))

        n = len(merged)
        rec = {
            "optimizer_A":     a,
            "optimizer_B":     b,
            "n_paired_seeds":  n,
            "mean_diff_AminusB": float(np.mean(
                merged[f"best_score_{a}"] - merged[f"best_score_{b}"]
            )) if n > 0 else np.nan,
        }

        if n >= 2 and (merged[f"best_score_{a}"] - merged[f"best_score_{b}"]).abs().sum() > 0:
            try:
                w = stats.wilcoxon(
                    merged[f"best_score_{a}"],
                    merged[f"best_score_{b}"],
                    zero_method="wilcox",
                    alternative="two-sided",
                    mode="exact" if n <= 25 else "auto",
                )
                rec["statistic"] = float(w.statistic)
                rec["p_value"]   = float(w.pvalue)
            except Exception as e:
                rec["statistic"] = np.nan
                rec["p_value"]   = np.nan
                rec["note"]      = f"wilcoxon failed: {e}"
        else:
            rec["statistic"] = np.nan
            rec["p_value"]   = np.nan
            rec["note"]      = "insufficient variation or seeds"

        rec["seeds"] = ",".join(map(str, sorted(merged["seed"].tolist())))
        rows.append(rec)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# P-value formatting for on-plot annotation
# -----------------------------------------------------------------------------
def format_pvalue(p):
    if p is None or (isinstance(p, float) and (np.isnan(p) or not np.isfinite(p))):
        return "p = n/a"
    if p < 1e-4:
        return "p < 1e-4"
    if p < 1e-3:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


def significance_stars(p):
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_convergence_mean_ci(conv_df, objective_col, out_png):
    plt.figure(figsize=(8, 5.0))
    optimizers = sorted(conv_df["optimizer"].unique())
    palette = dict(zip(optimizers, sns.color_palette("Set2", n_colors=len(optimizers))))

    for opt in optimizers:
        sub = conv_df[conv_df["optimizer"] == opt].sort_values("eval_index")
        c = palette[opt]
        plt.plot(sub["eval_index"], sub["mean"], linewidth=2.4, label=opt, color=c)
        plt.fill_between(
            sub["eval_index"], sub["ci95_low"], sub["ci95_high"],
            alpha=0.20, color=c,
        )

    plt.xlabel("Evaluation index")
    plt.ylabel(f"Best-so-far {objective_col} (mean ± 95% CI over seeds)")
    plt.title("Optimizer convergence across independent seeds")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_png.replace(".png", ".pdf"))
    plt.close()


def _lookup_pvalue(wilcoxon_df, opt_a, opt_b):
    if wilcoxon_df is None or wilcoxon_df.empty:
        return np.nan
    m = (
        ((wilcoxon_df["optimizer_A"] == opt_a) & (wilcoxon_df["optimizer_B"] == opt_b))
        | ((wilcoxon_df["optimizer_A"] == opt_b) & (wilcoxon_df["optimizer_B"] == opt_a))
    )
    hit = wilcoxon_df.loc[m]
    if hit.empty or "p_value" not in hit.columns:
        return np.nan
    return float(hit["p_value"].iloc[0])


def plot_best_score_boxplot(per_seed_df, objective_col, out_png, wilcoxon_df=None):
    """
    Boxplot of best_score per optimizer.

    Modifications relative to the earlier version:
      * No mean ± sd text above each box.
      * Manual scatter with deterministic per-seed x-offsets, so the same seed
        lands at the same relative x-position within every optimizer column.
      * Each point is annotated with its seed label, so the same seed can be
        visually tracked across scBOA_GP, Optuna_TPE, and Random_search.
    """
    order = (per_seed_df.groupby("optimizer")["best_score"].mean()
             .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    sns.boxplot(
        data=per_seed_df, x="optimizer", y="best_score",
        order=order, palette="Set2", width=0.5, showfliers=False, ax=ax,
    )

    # -------------------------------------------------------------------------
    # Deterministic per-seed x-offsets so seeds are trackable across optimizers
    # -------------------------------------------------------------------------
    seeds_sorted = sorted(per_seed_df["seed"].unique(), key=lambda s: str(s))
    n_seeds = len(seeds_sorted)
    if n_seeds > 1:
        offsets = np.linspace(-0.18, 0.18, n_seeds)
    else:
        offsets = np.array([0.0])
    seed_to_offset = dict(zip(seeds_sorted, offsets))

    # Assign a stable color per seed for extra visual tracking
    seed_palette = sns.color_palette("tab10", n_colors=max(n_seeds, 3))
    seed_to_color = {s: seed_palette[i % len(seed_palette)] for i, s in enumerate(seeds_sorted)}

    y_data_min = float(per_seed_df["best_score"].min())
    y_data_max = float(per_seed_df["best_score"].max())
    y_range    = max(y_data_max - y_data_min, 1e-6)
    label_dy   = 0.010 * y_range

    for i, opt in enumerate(order):
        sub = per_seed_df[per_seed_df["optimizer"] == opt]
        for _, row in sub.iterrows():
            seed = row["seed"]
            x = i + seed_to_offset[seed]
            y = float(row["best_score"])

            ax.scatter(
                x, y,
                s=42,
                color=seed_to_color[seed],
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
            ax.text(
                x, y + label_dy, f"seed {seed}",
                ha="center", va="bottom", fontsize=7.5, color="black",
                zorder=4,
            )

    # -------------------------------------------------------------------------
    # Pairwise Wilcoxon significance brackets
    # -------------------------------------------------------------------------
    if wilcoxon_df is not None and not wilcoxon_df.empty and len(order) >= 2:
        base = y_data_max + 0.09 * y_range
        step = 0.09 * y_range
        tick = 0.015 * y_range

        pairs = list(itertools.combinations(range(len(order)), 2))
        pairs.sort(key=lambda ij: (ij[1] - ij[0], ij[0]))

        for k, (i, j) in enumerate(pairs):
            opt_a, opt_b = order[i], order[j]
            p = _lookup_pvalue(wilcoxon_df, opt_a, opt_b)

            y = base + k * step

            ax.plot([i, i, j, j],
                    [y, y + tick, y + tick, y],
                    linewidth=1.2, color="black")

            label = f"{significance_stars(p)}   {format_pvalue(p)}"
            ax.text((i + j) / 2.0, y + tick * 1.2, label,
                    ha="center", va="bottom", fontsize=9)

        top_bracket = base + (len(pairs) - 1) * step + tick * 4
        cur_bottom, _ = ax.get_ylim()
        ax.set_ylim(cur_bottom, top_bracket)

    ax.set_ylabel(f"Best {objective_col} (per seed)")
    ax.set_xlabel("")
    ax.set_title("Between-seed variance in best objective score\n(pairwise Wilcoxon signed-rank)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_png.replace(".png", ".pdf"))
    plt.close()


def plot_param_variance(param_summary_df, out_png):
    if param_summary_df.empty:
        return
    df = param_summary_df.dropna(subset=["cv"]).copy()
    if df.empty:
        return

    plt.figure(figsize=(max(7, 0.55 * df["parameter"].nunique()), 4.8))
    sns.barplot(
        data=df, x="parameter", y="cv", hue="optimizer",
        palette="Set2",
    )
    plt.ylabel("Coefficient of variation of selected value (across seeds)")
    plt.xlabel("")
    plt.title("Between-seed variance in selected parameter configuration")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="optimizer", loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_png.replace(".png", ".pdf"))
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gp_stage_dirs", nargs="+", required=True,
        help="One stage_1_bayesian_optimization dir per seed for scBOA_GP")
    parser.add_argument("--random_stage_dirs", nargs="+", required=True,
        help="One stage_1_bayesian_optimization dir per seed for Random_search")
    parser.add_argument("--optuna_stage_dirs", nargs="+", default=None,
        help="One stage_1_bayesian_optimization dir per seed for Optuna_TPE")

    parser.add_argument("--seeds", nargs="+", required=True,
        help="Seed labels, aligned to the *_stage_dirs order (e.g. 43 44 45)")

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--objective_col", default=None)
    parser.add_argument("--max_trials", type=int, default=None,
        help="Equal-budget cap per (optimizer, seed) run, e.g. 90")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_seeds = len(args.seeds)
    if len(args.gp_stage_dirs) != n_seeds:
        raise ValueError("gp_stage_dirs length must match --seeds length.")
    if len(args.random_stage_dirs) != n_seeds:
        raise ValueError("random_stage_dirs length must match --seeds length.")
    if args.optuna_stage_dirs is not None and len(args.optuna_stage_dirs) != n_seeds:
        raise ValueError("optuna_stage_dirs length must match --seeds length.")

    optimizer_inputs = {
        "scBOA_GP":      args.gp_stage_dirs,
        "Random_search": args.random_stage_dirs,
    }
    if args.optuna_stage_dirs is not None:
        optimizer_inputs["Optuna_TPE"] = args.optuna_stage_dirs

    print("=" * 80)
    print(f"Seeds:      {args.seeds}")
    print(f"Optimizers: {list(optimizer_inputs.keys())}")
    print("=" * 80)

    if args.objective_col is None:
        peek_df = pd.read_csv(find_yield_report(args.gp_stage_dirs[0]))
        objective_col = infer_objective_column(peek_df)
        print(f"[Info] Inferred objective column: {objective_col}")
    else:
        objective_col = args.objective_col
        print(f"[Info] Using objective column: {objective_col}")

    all_runs = []
    per_seed_records  = []
    per_seed_params   = []

    for opt_name, stage_dirs in optimizer_inputs.items():
        for seed, stage_dir in zip(args.seeds, stage_dirs):
            run_df = load_run(
                stage_dir=stage_dir,
                optimizer_name=opt_name,
                seed=seed,
                objective_col=objective_col,
                max_trials=args.max_trials,
            )
            all_runs.append(run_df)

            summary, best_row = summarize_run(run_df)
            per_seed_records.append(summary)

            params = extract_best_params(best_row)
            params["optimizer"] = opt_name
            params["seed"]      = seed
            params["best_score"] = summary["best_score"]
            per_seed_params.append(params)

            print(f"  Loaded {opt_name} seed={seed}: "
                  f"n_trials={summary['n_trials']}, "
                  f"best={summary['best_score']:.5f}")

    all_runs_df = pd.concat(all_runs, ignore_index=True)
    per_seed_df = pd.DataFrame(per_seed_records)
    per_seed_params_df = pd.DataFrame(per_seed_params)

    per_seed_df.to_csv(os.path.join(args.output_dir, "per_seed_best_score.csv"), index=False)
    per_seed_params_df.to_csv(os.path.join(args.output_dir, "per_seed_best_params.csv"), index=False)

    best_score_summary = aggregate_best_score(per_seed_df)
    param_summary      = aggregate_params(per_seed_params_df)
    conv_df            = aggregate_convergence(all_runs_df)
    wilcoxon_df        = pairwise_wilcoxon(per_seed_df)

    best_score_summary.to_csv(
        os.path.join(args.output_dir, "optimizer_best_score_summary.csv"), index=False)
    param_summary.to_csv(
        os.path.join(args.output_dir, "optimizer_param_summary.csv"), index=False)
    conv_df.to_csv(
        os.path.join(args.output_dir, "convergence_mean_ci.csv"), index=False)
    wilcoxon_df.to_csv(
        os.path.join(args.output_dir, "pairwise_wilcoxon_best_score.csv"), index=False)

    plot_convergence_mean_ci(
        conv_df, objective_col,
        os.path.join(args.output_dir, "convergence_curve_mean_ci.png"),
    )
    plot_best_score_boxplot(
        per_seed_df, objective_col,
        os.path.join(args.output_dir, "best_score_boxplot.png"),
        wilcoxon_df=wilcoxon_df,
    )
    plot_param_variance(
        param_summary,
        os.path.join(args.output_dir, "param_variance_barplot.png"),
    )

    print("\n" + "=" * 80)
    print("Per-seed best scores")
    print("=" * 80)
    print(per_seed_df[["optimizer", "seed", "n_trials", "best_score",
                       "best_eval_index", "best_strategy"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("Best score summary (mean ± std across seeds)")
    print("=" * 80)
    print(best_score_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("Selected-parameter summary (mean ± std across seeds)")
    print("=" * 80)
    print(param_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("Pairwise Wilcoxon signed-rank on best_score")
    print("=" * 80)
    print(wilcoxon_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"All outputs saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()