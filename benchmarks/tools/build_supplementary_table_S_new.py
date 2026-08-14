#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Supplementary Table S_new: K-partial concordance audit (single GT granularity).

For each tissue and each external metric, report doubly-partial Spearman ρ
(controlling for cluster count + resolution). K* is defined as the top-5 median
cluster count ranked by scBOA final objective score.

Outputs:
  - supplementary_table_S_new.csv: main audit table (tissue × metric)
  - supplementary_table_S_new_kstar.csv: K* summary per tissue
"""

import argparse
import os
import sys
import pandas as pd
from scipy.stats import spearmanr


# Six external metrics cited in Response §2
EXTERNAL_METRICS = [
    "external_leiden_ari",
    "external_leiden_nmi",
    "external_homogeneity",                        # was external_leiden_homogeneity
    "external_completeness",
    "external_celltypist_broad_accuracy",
    "external_celltypist_broad_macro_f1",          # was external_celltypist_broad_f1_weighted
]


def load_tissue_data(run_dir, tissue_name):
    """Load the four CSVs from one tissue run directory and return unified dict."""
    trial_csv = os.path.join(run_dir, "trial_objective_facs_concordance.csv")
    spearman_csv = os.path.join(run_dir, "spearman_internal_vs_facs.csv")
    cluster_csv = os.path.join(run_dir, "cluster_number_vs_external_metrics_stats.csv")
    best_csv = os.path.join(run_dir, "best_vs_default_facs_concordance.csv")

    for f in [trial_csv, spearman_csv, cluster_csv, best_csv]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Missing {os.path.basename(f)}")

    trial_df = pd.read_csv(trial_csv).dropna(subset=["yield_score_target"])
    spearman_df = pd.read_csv(spearman_csv)
    cluster_df = pd.read_csv(cluster_csv)
    best_df = pd.read_csv(best_csv)

    # K* = top-5 median cluster count
    top5 = trial_df.nlargest(5, "yield_score_target")
    kstar = top5["n_leiden_clusters"].median()

    # B_eff from best_vs_default CSV
    b_eff = int(best_df["n_facs_labels"].iloc[0])

    # N_consensus from best trial in trial_df (ranked by yield_score_target)
    best_trial = trial_df.nlargest(1, "yield_score_target").iloc[0]
    n_consensus = int(best_trial["n_consensus_labels"])

    return {
        "tissue": tissue_name,
        "trial_df": trial_df,
        "spearman_df": spearman_df,
        "cluster_df": cluster_df,
        "best_df": best_df,
        "kstar": kstar,
        "b_eff": b_eff,
        "n_consensus": n_consensus,
    }


def compute_partial_correlation(trial_df, y_col, ctrl_cols):
    """
    Compute partial Spearman ρ(yield_score_target, y_col | ctrl_cols) via residuals.
    Returns (rho, p_value, n).
    """
    df = trial_df[["yield_score_target", y_col] + ctrl_cols].dropna()
    if len(df) < 10:
        return None, None, len(df)

    from sklearn.linear_model import LinearRegression

    X_ctrl = df[ctrl_cols].values
    y_target = df["yield_score_target"].values
    y_ext = df[y_col].values

    reg_target = LinearRegression().fit(X_ctrl, y_target)
    reg_ext = LinearRegression().fit(X_ctrl, y_ext)

    resid_target = y_target - reg_target.predict(X_ctrl)
    resid_ext = y_ext - reg_ext.predict(X_ctrl)

    rho, p = spearmanr(resid_target, resid_ext)
    return rho, p, len(df)


def build_table(tissues_data, output_dir):
    """Build two CSVs: main audit table + K* summary."""
    os.makedirs(output_dir, exist_ok=True)

    rows_main = []
    rows_kstar = []

    for tdata in tissues_data:
        tissue = tdata["tissue"]
        trial_df = tdata["trial_df"]
        spearman_df = tdata["spearman_df"]
        kstar = tdata["kstar"]
        b_eff = tdata["b_eff"]
        n_consensus = tdata["n_consensus"]

        print(f"\n{'='*70}")
        print(f"Tissue: {tissue.upper()}")
        print(f"  K* (top-5 median) = {kstar:.1f}")
        print(f"  B_eff = {b_eff}")
        print(f"  N_consensus = {n_consensus}")
        print(f"  K*/B_eff = {kstar/b_eff:.2f}")
        print(f"  N_consensus/B_eff = {n_consensus/b_eff:.2f}")
        print(f"{'='*70}")

        rows_kstar.append({
            "tissue": tissue,
            "K*": kstar,
            "B_eff": b_eff,
            "N_consensus": n_consensus,
            "K*/B_eff": kstar / b_eff if b_eff > 0 else None,
            "N_consensus/B_eff": n_consensus / b_eff if b_eff > 0 else None,
        })

        for ext_metric in EXTERNAL_METRICS:
            # Raw ρ from spearman_internal_vs_facs.csv
            row_match = spearman_df[
                (spearman_df["internal_metric"] == "yield_score_target") &
                (spearman_df["external_facs_metric"] == ext_metric)
            ]
            if row_match.empty:
                print(f"  ⚠️  {ext_metric}: not found in spearman CSV, skipping")
                continue

            raw_rho = row_match["spearman_rho"].iloc[0]
            raw_p = row_match["p_value"].iloc[0]

            # Partial ρ (ctrl K)
            partial_rho_K, partial_p_K, n_K = compute_partial_correlation(
                trial_df, ext_metric, ["n_leiden_clusters"]
            )

            # Doubly-partial ρ (ctrl K + resolution)
            partial_rho_KR, partial_p_KR, n_KR = compute_partial_correlation(
                trial_df, ext_metric, ["n_leiden_clusters", "resolution"]
            )

            print(f"\n  {ext_metric}:")
            print(f"    Raw ρ           = {raw_rho:+.3f}  (p={raw_p:.4f})")
            print(f"    Partial ρ | K   = {partial_rho_K:+.3f}  (n={n_K})")
            print(f"    Partial ρ | K,R = {partial_rho_KR:+.3f}  (n={n_KR})")

            rows_main.append({
                "tissue": tissue,
                "external_metric": ext_metric,
                "B_eff": b_eff,
                "N_consensus": n_consensus,
                "K*": kstar,
                "raw_rho": raw_rho,
                "raw_p": raw_p,
                "partial_rho_ctrl_K": partial_rho_K,
                "partial_rho_ctrl_K_resolution": partial_rho_KR,
                "n_trials": n_KR,
            })

    # Write main table
    df_main = pd.DataFrame(rows_main)
    main_path = os.path.join(output_dir, "supplementary_table_S_new.csv")
    df_main.to_csv(main_path, index=False, float_format="%.4f")
    print(f"\n{'='*70}")
    print(f"✅ Main audit table saved to:\n   {main_path}")
    print(f"   Shape: {df_main.shape}")

    # Write K* summary
    df_kstar = pd.DataFrame(rows_kstar)
    kstar_path = os.path.join(output_dir, "supplementary_table_S_new_kstar.csv")
    df_kstar.to_csv(kstar_path, index=False, float_format="%.4f")
    print(f"✅ K* summary saved to:\n   {kstar_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Build Supplementary Table S_new")
    parser.add_argument(
        "--run",
        nargs="+",
        required=True,
        metavar="TISSUE:PATH",
        help="One or more tissue:path pairs, e.g., pbmc:/path/to/pbmc/objective_to_facs_concordance",
    )
    parser.add_argument(
        "--output_dir",
        default="./supplementary_table_S_new_output",
        help="Output directory for CSVs",
    )
    args = parser.parse_args()

    tissues_data = []
    for run_spec in args.run:
        if ":" not in run_spec:
            print(f"❌ Invalid format: {run_spec}. Expected TISSUE:PATH")
            sys.exit(1)
        tissue_name, run_dir = run_spec.split(":", 1)
        run_dir = os.path.expanduser(run_dir)
        print(f"Loading {tissue_name} from {run_dir}...")
        try:
            tdata = load_tissue_data(run_dir, tissue_name)
            tissues_data.append(tdata)
            print(f"   ✅ Loaded {len(tdata['trial_df'])} trials")
        except Exception as e:
            print(f"   ❌ Failed to load {tissue_name}: {e}")
            sys.exit(1)

    build_table(tissues_data, args.output_dir)


if __name__ == "__main__":
    main()