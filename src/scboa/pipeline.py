#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated Two-Stage Bayesian Optimization and Final Analysis Pipeline for scRNA-seq.

This script combines a Bayesian optimization stage for parameter discovery with a
final, detailed analysis stage that uses the discovered optimal parameters. It now
supports iterative refinement for low-confidence cells.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import time
import celltypist
from celltypist import models
import argparse
import random
import re
import anndata
import sys
import matplotlib

# Added import for robust marker aggregation in Stage 2
from collections import defaultdict

# Use 'Agg' backend for non-interactive environments
matplotlib.use('Agg')

# --- Conditional Import for Harmony ---
try:
    import harmonypy as hm
except ImportError:
    print("Warning: harmonypy is not installed. Multi-sample integration mode will not be available.")
    print("Please run 'pip install harmonypy'")


# --- Bayesian Optimization Imports ---
try:
    from skopt import gp_minimize, dump, load
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args
    from skopt.plots import plot_evaluations, plot_objective
except ImportError:
    print("Error: scikit-optimize is not installed. Please run 'pip install scikit-optimize'")
    exit()

# --- Visualization Imports ---
try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import umap
except ImportError:
    print("Warning: Matplotlib, Seaborn, Scikit-learn, or UMAP not installed. Visualization feature will not work.")
    print("Please run 'pip install matplotlib seaborn scikit-learn umap-learn'")


# ==============================================================================
# --- *** CONFIGURATION SECTION *** ---
# ==============================================================================
MITO_REGEX_PATTERN = r'^(MT|Mt|mt)[-._:]'

# Default search space for Stage 1, 'n_hvg' may be dynamically changed later
search_space = [
    Integer(200, 20000, name='n_hvg'),
    Integer(10, 100, name='n_pcs'),
    Integer(10, 50, name='n_neighbors'),
    Real(0.2, 2.0, name='resolution')
]

# --- Global variables for Stage 1 ---
adata_base = None
model = None
RANDOM_SEED = None
ARGS = None  # Will hold parsed command-line arguments
CURRENT_OPTIMIZATION_TARGET = None
CURRENT_STRATEGY_NAME = ""
TRIAL_METADATA = [] # Holds per-trial metadata (e.g., scores, label counts)


# ==============================================================================
# ==============================================================================
# --- *** STAGE 1: BAYESIAN OPTIMIZATION FUNCTIONS *** ---
# ==============================================================================
# ==============================================================================
@use_named_args(dimensions=search_space)
def objective_function(n_hvg, n_pcs, n_neighbors, resolution):
    """
    (Stage 1) Runs the appropriate pipeline (single-sample or integrated), calculates all
    metrics (CAS, MCS, Silhouette), and returns a score based on the global
    CURRENT_OPTIMIZATION_TARGET.
    """
    global adata_base, model, RANDOM_SEED, ARGS, CURRENT_OPTIMIZATION_TARGET, CURRENT_STRATEGY_NAME, TRIAL_METADATA

    print(f"\n---> [{CURRENT_STRATEGY_NAME}] Trial: HVGs={n_hvg}, PCs={n_pcs}, Neighbors={n_neighbors}, Resolution={resolution:.3f}")
    start_time = time.time()

    adata_proc = adata_base.copy()
    is_multi_sample = 'sample' in adata_base.obs.columns

    # Use the raw layer for annotation if it exists, otherwise use .X
    adata_for_annot = adata_proc.raw.to_adata() if adata_proc.raw is not None else adata_proc
    print("     [INFO] Annotating individual cells on full log-normalized data...")
    predictions = celltypist.annotate(adata_for_annot, model=model, majority_voting=False)
    adata_proc.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels']

    is_two_step_hvg = all(p is not None for p in [ARGS.hvg_min_mean, ARGS.hvg_max_mean, ARGS.hvg_min_disp])
    if is_two_step_hvg:
        print("     [INFO] Trial using two-step sequential HVG selection.")
        sc.pp.highly_variable_genes(
            adata_proc,
            min_mean=ARGS.hvg_min_mean,
            max_mean=ARGS.hvg_max_mean,
            min_disp=ARGS.hvg_min_disp,
            batch_key='sample' if is_multi_sample else None
        )
        hvg_df = adata_proc.var[adata_proc.var.highly_variable]
        hvg_df = hvg_df.sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:n_hvg]
        adata_proc.var['highly_variable'] = False
        adata_proc.var.loc[top_genes, 'highly_variable'] = True
    else:
        print("     [INFO] Trial using rank-based HVG selection.")
        if is_multi_sample:
            sc.pp.highly_variable_genes(adata_proc, n_top_genes=n_hvg, batch_key='sample', flavor='seurat_v3')
        else:
            sc.pp.highly_variable_genes(adata_proc, n_top_genes=n_hvg, flavor='seurat_v3')

    adata_proc = adata_proc[:, adata_proc.var.highly_variable].copy()
    sc.pp.scale(adata_proc, max_value=10)
    
    # Cap n_pcs_compute by the number of available features
    n_pcs_to_compute = min(ARGS.n_pcs_compute, adata_proc.n_obs - 1, adata_proc.n_vars - 1)
    if n_pcs > n_pcs_to_compute:
        print(f"     [WARNING] Requested n_pcs ({n_pcs}) > computed PCs ({n_pcs_to_compute}). Capping at {n_pcs_to_compute}.")
        n_pcs = n_pcs_to_compute

    sc.tl.pca(adata_proc, svd_solver='arpack', n_comps=n_pcs_to_compute, random_state=RANDOM_SEED)

    embedding_to_use = 'X_pca'
    if is_multi_sample:
        sc.external.pp.harmony_integrate(
            adata_proc,
            key='sample',
            basis='X_pca',
            adjusted_basis='X_pca_harmony',
            random_state=RANDOM_SEED
        )
        embedding_to_use = 'X_pca_harmony'

    sc.pp.neighbors(adata_proc, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=embedding_to_use, random_state=RANDOM_SEED)
    sc.tl.leiden(adata_proc, resolution=resolution, random_state=RANDOM_SEED)

    silhouette_avg = 0.0
    rescaled_silhouette = 0.0
    try:
        n_clusters = adata_proc.obs['leiden'].nunique()
        if n_clusters > 1:
            silhouette_avg = silhouette_score(adata_proc.obsm[embedding_to_use][:, :n_pcs], adata_proc.obs['leiden'])
            rescaled_silhouette = (silhouette_avg + 1) / 2
        else:
            silhouette_avg = -1.0; rescaled_silhouette = 0.0
    except Exception as e:
        print(f"     [WARNING] Could not calculate silhouette score. Error: {e}. Scores set to worst values.")
        silhouette_avg = -1.0; rescaled_silhouette = 0.0

    cluster2label = adata_proc.obs.groupby('leiden')['ctpt_individual_prediction'].agg(lambda x: x.value_counts().idxmax())
    adata_proc.obs['ctpt_consensus_prediction'] = adata_proc.obs['leiden'].map(cluster2label)
    total_cells = len(adata_proc.obs)
    total_matching = (adata_proc.obs['ctpt_individual_prediction'] == adata_proc.obs['ctpt_consensus_prediction']).sum()
    weighted_mean_cas = (total_matching / total_cells) * 100 if total_cells > 0 else 0.0

    simple_mean_cas = 0.0
    if ARGS.cas_aggregation_method == 'leiden':
        cas_per_cluster = [
            g['ctpt_individual_prediction'].eq(g['ctpt_consensus_prediction'].iloc[0]).mean() * 100
            for _, g in adata_proc.obs.groupby('leiden') if not g.empty
        ]
        simple_mean_cas = np.mean(cas_per_cluster) if cas_per_cluster else 0.0
    elif ARGS.cas_aggregation_method == 'consensus':
        cas_per_consensus_group = [
            g['ctpt_individual_prediction'].eq(g['ctpt_consensus_prediction'].iloc[0]).mean() * 100
            for _, g in adata_proc.obs.groupby('ctpt_consensus_prediction') if not g.empty
        ]
        simple_mean_cas = np.mean(cas_per_consensus_group) if cas_per_consensus_group else 0.0

    mean_mcs = 0.0
    try:
        label_counts = adata_proc.obs['ctpt_consensus_prediction'].value_counts()
        valid_labels = label_counts[label_counts > 1].index.tolist()
        if len(valid_labels) >= 2:
            sc.tl.rank_genes_groups(
                adata_proc, 'ctpt_consensus_prediction', groups=valid_labels,
                method='wilcoxon', use_raw=True, key_added='rank_genes_consensus'
            )
            marker_df = sc.get.rank_genes_groups_df(adata_proc, key='rank_genes_consensus', group=None)
            is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
            if ARGS.marker_gene_model == 'non-mitochondrial':
                filtered_rows = [sub[~sub['names'].map(is_mito)].head(ARGS.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            else:
                filtered_rows = [sub.head(ARGS.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            top_genes_per_group = pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame()

            if not top_genes_per_group.empty:
                unique_top_genes = top_genes_per_group['names'].unique().tolist()
                data_df = sc.get.obs_df(adata_proc, keys=['ctpt_consensus_prediction'] + unique_top_genes, use_raw=True)
                fraction_df = data_df.groupby('ctpt_consensus_prediction').apply(lambda x: (x[unique_top_genes] > 0).mean())
                mcs_scores = {cell_type: fraction_df.loc[cell_type, top_genes_per_group[top_genes_per_group['group'] == cell_type]['names']].mean() for cell_type in top_genes_per_group['group'].unique()}
                if mcs_scores: mean_mcs = np.mean(list(mcs_scores.values())) * 100
    except Exception as e:
        print(f"     [WARNING] Could not calculate MCS for this trial. Error: {e}. MCS set to 0.")
        mean_mcs = 0.0

    TRIAL_METADATA.append({
        'n_individual_labels': adata_proc.obs['ctpt_individual_prediction'].nunique(),
        'n_consensus_labels': adata_proc.obs['ctpt_consensus_prediction'].nunique(),
        'weighted_mean_cas': weighted_mean_cas, 'simple_mean_cas': simple_mean_cas,
        'mean_mcs': mean_mcs, 'silhouette_score_original': silhouette_avg, 'silhouette_score_rescaled': rescaled_silhouette
    })
    end_time = time.time()

    if CURRENT_OPTIMIZATION_TARGET == 'weighted_cas':
        score = weighted_mean_cas
    elif CURRENT_OPTIMIZATION_TARGET == 'simple_cas':
        score = simple_mean_cas
    elif CURRENT_OPTIMIZATION_TARGET == 'mcs':
        score = mean_mcs
    elif CURRENT_OPTIMIZATION_TARGET == 'balanced':
        epsilon = 1e-6
        if ARGS.model_type == 'structural':
            score = (((weighted_mean_cas / 100 + epsilon) * (simple_mean_cas / 100 + epsilon) * (mean_mcs / 100 + epsilon) * (rescaled_silhouette + epsilon)) ** (1/4.0)) * 100
        elif ARGS.model_type == 'silhouette':
            score = silhouette_avg
        else: # 'biological' model
            score = (((weighted_mean_cas / 100 + epsilon) * (simple_mean_cas / 100 + epsilon) * (mean_mcs / 100 + epsilon)) ** (1/3.0)) * 100
    else:
        raise ValueError(f"Invalid optimization target: '{CURRENT_OPTIMIZATION_TARGET}'")

    print(f"<--- Results (Time: {end_time - start_time:.1f}s) -> Score: {score:.3f}")
    return -score

def evaluate_final_metrics(params_dict):
    """(Stage 1) Runs the appropriate pipeline once to get final metrics and the AnnData object for saving."""
    print("\n--- Re-running analysis with overall best parameters for final report ---")
    adata_final = adata_base.copy()
    is_multi_sample = 'sample' in adata_base.obs.columns

    adata_for_annot = adata_final.raw.to_adata() if adata_final.raw is not None else adata_final
    print("     [INFO] Final run: Annotating individual cells on full log-normalized data...")
    predictions = celltypist.annotate(adata_for_annot, model=model, majority_voting=False)
    adata_final.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels']

    is_two_step_hvg = all(p is not None for p in [ARGS.hvg_min_mean, ARGS.hvg_max_mean, ARGS.hvg_min_disp])
    if is_two_step_hvg:
        print("     [INFO] Final run using two-step sequential HVG selection.")
        sc.pp.highly_variable_genes(adata_final, min_mean=ARGS.hvg_min_mean, max_mean=ARGS.hvg_max_mean, min_disp=ARGS.hvg_min_disp, batch_key='sample' if is_multi_sample else None)
        hvg_df = adata_final.var[adata_final.var.highly_variable].sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:params_dict['n_hvg']]
        adata_final.var['highly_variable'] = False
        adata_final.var.loc[top_genes, 'highly_variable'] = True
    else:
        print("     [INFO] Final run using rank-based HVG selection.")
        if is_multi_sample:
            sc.pp.highly_variable_genes(adata_final, n_top_genes=params_dict['n_hvg'], batch_key='sample', flavor='seurat_v3')
        else:
            sc.pp.highly_variable_genes(adata_final, n_top_genes=params_dict['n_hvg'], flavor='seurat_v3')

    adata_final = adata_final[:, adata_final.var.highly_variable].copy()
    sc.pp.scale(adata_final, max_value=10)
    
    n_pcs_to_compute = min(ARGS.n_pcs_compute, adata_final.n_obs - 1, adata_final.n_vars - 1)
    n_pcs = min(params_dict['n_pcs'], n_pcs_to_compute)
    sc.tl.pca(adata_final, svd_solver='arpack', n_comps=n_pcs_to_compute, random_state=RANDOM_SEED)

    embedding_to_use = 'X_pca'
    if is_multi_sample:
        sc.external.pp.harmony_integrate(adata_final, key='sample', basis='X_pca', adjusted_basis='X_pca_harmony', random_state=RANDOM_SEED)
        embedding_to_use = 'X_pca_harmony'

    sc.pp.neighbors(adata_final, n_neighbors=params_dict['n_neighbors'], n_pcs=n_pcs, use_rep=embedding_to_use, random_state=RANDOM_SEED)
    sc.tl.leiden(adata_final, resolution=params_dict['resolution'], random_state=RANDOM_SEED)
    sc.tl.umap(adata_final, random_state=RANDOM_SEED)

    silhouette_avg, rescaled_silhouette = 0.0, 0.0
    try:
        if adata_final.obs['leiden'].nunique() > 1:
            silhouette_avg = silhouette_score(adata_final.obsm[embedding_to_use][:, :n_pcs], adata_final.obs['leiden'])
            rescaled_silhouette = (silhouette_avg + 1) / 2
        else:
            silhouette_avg = -1.0; rescaled_silhouette = 0.0
    except Exception as e:
        print(f"[WARNING] Final silhouette calculation failed. Error: {e}. Scores set to worst values.")
        silhouette_avg = -1.0; rescaled_silhouette = 0.0

    cluster2label = adata_final.obs.groupby('leiden')['ctpt_individual_prediction'].agg(lambda x: x.value_counts().idxmax())
    adata_final.obs['ctpt_consensus_prediction'] = adata_final.obs['leiden'].map(cluster2label)
    total_cells, total_matching = len(adata_final.obs), (adata_final.obs['ctpt_individual_prediction'] == adata_final.obs['ctpt_consensus_prediction']).sum()
    weighted_cas = (total_matching / total_cells) * 100 if total_cells > 0 else 0.0

    simple_cas = 0.0
    if ARGS.cas_aggregation_method == 'leiden':
        cas_per_cluster = [g['ctpt_individual_prediction'].eq(g['ctpt_consensus_prediction'].iloc[0]).mean() * 100 for _, g in adata_final.obs.groupby('leiden') if not g.empty]
        simple_cas = np.mean(cas_per_cluster) if cas_per_cluster else 0.0
    elif ARGS.cas_aggregation_method == 'consensus':
        cas_per_consensus_group = [g['ctpt_individual_prediction'].eq(g['ctpt_consensus_prediction'].iloc[0]).mean() * 100 for _, g in adata_final.obs.groupby('ctpt_consensus_prediction') if not g.empty]
        simple_cas = np.mean(cas_per_consensus_group) if cas_per_consensus_group else 0.0

    mean_mcs = 0.0
    try:
        label_counts = adata_final.obs['ctpt_consensus_prediction'].value_counts()
        valid_labels = label_counts[label_counts > 1].index.tolist()
        if len(valid_labels) >= 2:
            sc.tl.rank_genes_groups(adata_final, 'ctpt_consensus_prediction', groups=valid_labels, method='wilcoxon', use_raw=True, key_added='rank_genes_consensus')
            marker_df = sc.get.rank_genes_groups_df(adata_final, key='rank_genes_consensus', group=None)
            is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
            if ARGS.marker_gene_model == 'non-mitochondrial':
                filtered_rows = [sub[~sub['names'].map(is_mito)].head(ARGS.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            else:
                filtered_rows = [sub.head(ARGS.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            top_genes_per_group = pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame()
            if not top_genes_per_group.empty:
                unique_top_genes = top_genes_per_group['names'].unique().tolist()
                data_df = sc.get.obs_df(adata_final, keys=['ctpt_consensus_prediction'] + unique_top_genes, use_raw=True)
                fraction_df = data_df.groupby('ctpt_consensus_prediction').apply(lambda x: (x[unique_top_genes] > 0).mean())
                mcs_scores = {cell_type: fraction_df.loc[cell_type, top_genes_per_group[top_genes_per_group['group'] == cell_type]['names']].mean() for cell_type in top_genes_per_group['group'].unique()}
                if mcs_scores: mean_mcs = np.mean(list(mcs_scores.values())) * 100
    except Exception as e:
        print(f"[WARNING] Final MCS calculation failed. Error: {e}. MCS set to 0.")
        mean_mcs = 0.0

    epsilon = 1e-6
    if ARGS.model_type == 'structural':
        balanced_score = (((weighted_cas / 100 + epsilon) * (simple_cas / 100 + epsilon) * (mean_mcs / 100 + epsilon) * (rescaled_silhouette + epsilon)) ** (1/4.0)) * 100
    elif ARGS.model_type == 'silhouette':
        balanced_score = silhouette_avg
    else: # 'biological' model
        balanced_score = (((weighted_cas / 100 + epsilon) * (simple_cas / 100 + epsilon) * (mean_mcs / 100 + epsilon)) ** (1/3.0)) * 100

    return {
        "weighted_mean_cas": weighted_cas, "simple_mean_cas": simple_cas, "mean_mcs": mean_mcs,
        "silhouette_score_original": silhouette_avg, "rescaled_silhouette_score": rescaled_silhouette,
        "balanced_score": balanced_score, "n_individual_labels": adata_final.obs['ctpt_individual_prediction'].nunique(),
        "n_consensus_labels": adata_final.obs['ctpt_consensus_prediction'].nunique()
    }, adata_final
def print_final_report(target_name, params, metrics, winning_strategy):
    """(Stage 1) Prints a formatted final report to the console."""
    target_title_map = {'weighted_cas': "Weighted Mean CAS", 'simple_cas': "Simple Mean CAS", 'mcs': "Mean MCS", 'balanced': "Balanced Score (CAS & MCS)"}
    if ARGS.model_type == 'structural': target_title_map['balanced'] = "Balanced Score (CAS, MCS & Silhouette)"
    elif ARGS.model_type == 'silhouette': target_title_map['balanced'] = "Silhouette Score"
    target_title = target_title_map.get(target_name, "Unknown Target")
    print("\n" + "="*60 + f"\n--- Final Report for {target_title} Optimization ---\n--- (Best result found by '{winning_strategy}' strategy) ---\n\n--- Optimal Parameters Found ---")
    for key, value in params.items(): print(f"  - Best {key}: {value:.3f}" if isinstance(value, float) else f"  - Best {key}: {value}")
    print("\n--- Final Metrics for Optimal Parameters ---")
    if target_name == 'balanced':
        format_str = ".3f" if ARGS.model_type == 'silhouette' else ".2f"
        print(f"  - Highest {target_title}: {metrics['balanced_score']:{format_str}}")
    elif target_name == 'weighted_cas': print(f"  - Highest Weighted Mean CAS: {metrics['weighted_mean_cas']:.2f}%")
    elif target_name == 'simple_cas': print(f"  - Highest Simple Mean CAS: {metrics['simple_mean_cas']:.2f}%")
    elif target_name == 'mcs': print(f"  - Highest Mean MCS: {metrics['mean_mcs']:.2f}%")

    print(f"  - Corresponding Weighted Mean CAS: {metrics['weighted_mean_cas']:.2f}%\n  - Corresponding Simple Mean CAS: {metrics['simple_mean_cas']:.2f}%\n  - Corresponding Mean MCS: {metrics['mean_mcs']:.2f}%\n  - Corresponding Silhouette Score: {metrics['silhouette_score_original']:.3f}\n  - Final # of individual cell labels: {metrics['n_individual_labels']}\n  - Final # of consensus cluster labels: {metrics['n_consensus_labels']}\n" + "="*60)
def save_results_to_file(output_path, target_name, params, metrics, winning_strategy):
    """(Stage 1) Saves the final report to a text file."""
    target_title_map = {'weighted_cas': "Weighted Mean CAS", 'simple_cas': "Simple Mean CAS", 'mcs': "Mean MCS", 'balanced': "Balanced Score (Geometric Mean of CAS & MCS)"}
    if ARGS.model_type == 'structural': target_title_map['balanced'] = "Balanced Score (Geometric Mean of CAS, MCS & Silhouette)"
    elif ARGS.model_type == 'silhouette': target_title_map['balanced'] = "Silhouette Score"
    target_title = target_title_map.get(target_name, "Unknown Target")
    with open(output_path, 'w') as f:
        f.write(f"--- Bayesian Optimization Results ---\nOptimization Model Type: {ARGS.model_type}\nMarker Gene Model for MCS: {ARGS.marker_gene_model}\nOptimization Target: {target_title}\nWinning Strategy: {winning_strategy}\nRandom Seed Used: {RANDOM_SEED}\n\n")
        for key, value in params.items(): f.write(f"Best {key}: {value:.3f}\n" if isinstance(value, float) else f"Best {key}: {value}\n")
        f.write("\n")
        if target_name == 'balanced':
            if ARGS.model_type == 'silhouette':
                f.write(f"Highest_silhouette_score: {metrics['balanced_score']:.4f}\n")
            else:
                f.write(f"Highest_balanced_score: {metrics['balanced_score']:.4f}\n")
        elif target_name == 'weighted_cas': f.write(f"Highest_weighted_mean_cas_pct: {metrics['weighted_mean_cas']:.2f}\n")
        elif target_name == 'simple_cas': f.write(f"Highest_simple_mean_cas_pct: {metrics['simple_mean_cas']:.2f}\n")
        elif target_name == 'mcs': f.write(f"Highest_mean_mcs_pct: {metrics['mean_mcs']:.2f}\n")
        f.write(f"Corresponding_weighted_mean_cas_pct: {metrics['weighted_mean_cas']:.2f}\nCorresponding_simple_mean_cas_pct: {metrics['simple_mean_cas']:.2f}\nCorresponding_mean_mcs_pct: {metrics['mean_mcs']:.2f}\nCorresponding_silhouette_score: {metrics['silhouette_score_original']:.4f}\nFinal_n_individual_labels: {metrics['n_individual_labels']}\nFinal_n_consensus_labels: {metrics['n_consensus_labels']}\n")
def generate_yield_csv(results_dict, target_metric, output_dir, output_prefix):
    """(Stage 1) Generates a consolidated CSV file with detailed results from all trials."""
    print("\n--- Generating consolidated yield CSV report ---")
    param_names = ['n_hvg', 'n_pcs', 'n_neighbors', 'resolution']
    all_dfs = []
    for name, result in results_dict.items():
        params_df = pd.DataFrame(result.x_iters, columns=param_names)
        if hasattr(result, 'trial_metadata') and len(result.trial_metadata) == len(params_df):
            metadata_df = pd.DataFrame(result.trial_metadata)
            base_df = pd.concat([params_df, metadata_df], axis=1)
        else:
            print(f"  [WARNING] Per-trial metadata not found for strategy '{name}'. Metric/label columns will be empty.")
            base_df = params_df.copy()
            for col in ['n_individual_labels', 'n_consensus_labels', 'weighted_mean_cas', 'simple_mean_cas', 'mean_mcs', 'silhouette_score_original', 'silhouette_score_rescaled']:
                base_df[col] = np.nan
        base_df['yield_score_target'], base_df['call_number'], base_df['strategy'] = -np.array(result.func_vals), range(1, len(result.func_vals) + 1), name
        all_dfs.append(base_df)
    if not all_dfs: print("  [ERROR] No results found to generate CSV. Skipping."); return
    final_df = pd.concat(all_dfs, ignore_index=True)
    epsilon = 1e-6

    required_cols = ['weighted_mean_cas', 'simple_mean_cas', 'mean_mcs', 'silhouette_score_rescaled', 'silhouette_score_original']
    if all(col in final_df.columns for col in required_cols):
        if ARGS.model_type == 'structural':
            final_df['balanced_score_gmean'] = (((final_df['weighted_mean_cas'].fillna(0) / 100 + epsilon) *
                                                 (final_df['simple_mean_cas'].fillna(0) / 100 + epsilon) *
                                                 (final_df['mean_mcs'].fillna(0) / 100 + epsilon) *
                                                 (final_df['silhouette_score_rescaled'].fillna(0) + epsilon)) ** (1/4.0)) * 100
        elif ARGS.model_type == 'silhouette':
            final_df['balanced_score_gmean'] = final_df['silhouette_score_original']
        else: # 'biological' model
            final_df['balanced_score_gmean'] = (((final_df['weighted_mean_cas'].fillna(0) / 100 + epsilon) *
                                                 (final_df['simple_mean_cas'].fillna(0) / 100 + epsilon) *
                                                 (final_df['mean_mcs'].fillna(0) / 100 + epsilon)) ** (1/3.0)) * 100
    else:
        final_df['balanced_score_gmean'] = np.nan
    final_df.rename(columns={'silhouette_score_original': 'silhouette_score'}, inplace=True)
    final_column_order = ['call_number', 'strategy', 'n_hvg', 'n_pcs', 'n_neighbors', 'resolution', 'yield_score_target', 'balanced_score_gmean', 'weighted_mean_cas', 'simple_mean_cas', 'mean_mcs', 'silhouette_score', 'n_individual_labels', 'n_consensus_labels']
    final_df = final_df.reindex(columns=final_column_order)
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_yield_scores_report.csv")
    final_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"✅ Success! Saved consolidated CSV report to: {output_path}")
def plot_optimizer_paths_tsne(results, target_metric, output_dir, output_prefix, n_points_to_show=25):
    """(Stage 1) Generates a t-SNE plot of the search space with publication-quality styling."""
    print("\n--- Generating t-SNE visualization with publication-quality style ---")
    all_points = np.array(list(set(tuple(p) for res in results.values() for p in res.x_iters)))
    if len(all_points) <= 1: print("Skipping t-SNE plot: not enough unique points to embed."); return
    print(f"Found {len(all_points)} unique points. Performing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_points) - 1), random_state=RANDOM_SEED, max_iter=1000, init='pca', learning_rate='auto')
    tsne_coords_map = {tuple(p): tsne_coord for p, tsne_coord in zip(all_points, tsne.fit_transform(StandardScaler().fit_transform(all_points)))}
    all_tsne_coords = np.array(list(tsne_coords_map.values()))
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(12, 10)); ax.grid(False)
    cluster_labels = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init='auto').fit_predict(all_tsne_coords)
    ax.scatter(all_tsne_coords[:, 0], all_tsne_coords[:, 1], c=cluster_labels, cmap='tab10', alpha=0.2, s=80, zorder=1)
    colors = {'Exploit': '#d62728', 'BO-EI': "#fcbe06", 'Explore': "#9015d2"}
    for name, result in results.items():
        if name in colors:
            path_coords = np.array([tsne_coords_map[tuple(p)] for p in result.x_iters[:n_points_to_show]])
            for i in range(len(path_coords) - 1): ax.annotate('', xy=path_coords[i+1], xytext=path_coords[i], arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.5", color=colors[name], lw=2.0, shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0.2"), zorder=3)
    legend = ax.legend(handles=[Line2D([0], [0], label=name, color=color, linestyle='-', linewidth=4) for name, color in colors.items() if name in results], title='Strategy', fontsize=28, loc='best', title_fontsize=28)
    legend.get_title().set_fontweight('bold'); [text.set_fontweight('bold') for text in legend.get_texts()]
    ax.set_title(f"Optimizer Paths (First {n_points_to_show} Steps)\nTarget: {target_metric.replace('_', ' ').title()}", fontsize=28, fontweight='bold'); ax.set_xlabel("t-SNE 1", fontsize=28, fontweight='bold'); ax.set_ylabel("t-SNE 2", fontsize=28, fontweight='bold'); ax.tick_params(axis='both', which='major', labelsize=28, width=1.2)
    [label.set_fontweight('bold') for label in ax.get_xticklabels() + ax.get_yticklabels()]
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_optimizer_paths_tsne.png")
    plt.savefig(output_path, dpi=500, bbox_inches='tight'); plt.close()
    print(f"✅ Success! Saved t-SNE plot to: {output_path}")
def plot_optimizer_paths_umap(results, target_metric, output_dir, output_prefix, n_points_to_show=25):
    """(Stage 1) Generates a UMAP plot of the search space with publication-quality styling."""
    print("\n--- Generating UMAP visualization with publication-quality style ---")
    all_points = np.array(list(set(tuple(p) for res in results.values() for p in res.x_iters)))
    if len(all_points) <= 1: print("Skipping UMAP plot: not enough unique points to embed."); return
    print(f"Found {len(all_points)} unique points. Performing UMAP embedding...")

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
    umap_coords_map = {tuple(p): umap_coord for p, umap_coord in zip(all_points, reducer.fit_transform(StandardScaler().fit_transform(all_points)))}
    all_umap_coords = np.array(list(umap_coords_map.values()))

    plt.style.use('seaborn-v0_8-white'); fig, ax = plt.subplots(figsize=(12, 10)); ax.grid(False)
    cluster_labels = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init='auto').fit_predict(all_umap_coords)
    ax.scatter(all_umap_coords[:, 0], all_umap_coords[:, 1], c=cluster_labels, cmap='tab10', alpha=0.2, s=80, zorder=1)
    colors = {'Exploit': '#d62728', 'BO-EI': "#fcbe06", 'Explore': "#9015d2"}

    for name, result in results.items():
        if name in colors:
            path_coords = np.array([umap_coords_map[tuple(p)] for p in result.x_iters[:n_points_to_show]])
            for i in range(len(path_coords) - 1): ax.annotate('', xy=path_coords[i+1], xytext=path_coords[i], arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.5", color=colors[name], lw=2.0, shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0.2"), zorder=3)
    
    legend = ax.legend(handles=[Line2D([0], [0], label=name, color=color, linestyle='-', linewidth=4) for name, color in colors.items() if name in results], title='Strategy', fontsize=28, loc='best', title_fontsize=28)
    legend.get_title().set_fontweight('bold'); [text.set_fontweight('bold') for text in legend.get_texts()]
    ax.set_title(f"Optimizer Paths (First {n_points_to_show} Steps)\nTarget: {target_metric.replace('_', ' ').title()}", fontsize=28, fontweight='bold')
    ax.set_xlabel("UMAP 1", fontsize=28, fontweight='bold'); ax.set_ylabel("UMAP 2", fontsize=28, fontweight='bold'); ax.tick_params(axis='both', which='major', labelsize=28, width=1.2)
    [label.set_fontweight('bold') for label in ax.get_xticklabels() + ax.get_yticklabels()]
    
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_optimizer_paths_umap.png")
    plt.savefig(output_path, dpi=500, bbox_inches='tight'); plt.close()
    print(f"✅ Success! Saved UMAP plot to: {output_path}")
def plot_optimizer_convergence(results, target_metric, output_dir, output_prefix):
    """(Stage 1) Generates a convergence plot with publication-quality styling."""
    print("\n--- Generating convergence plot with publication-quality style ---")
    plt.style.use('seaborn-v0_8-white'); fig, ax = plt.subplots(figsize=(22, 10)); ax.grid(False)
    colors, font_size, max_x = {'Exploit': '#d62728', 'BO-EI': "#fcbe06", 'Explore': "#9015d2"}, 28, 0
    for name, result in results.items():
        if name in colors:
            best_so_far = np.maximum.accumulate(-np.array(result.func_vals))
            x = np.arange(1, len(best_so_far) + 1); max_x = max(max_x, x.max())
            ax.plot(x, best_so_far, marker='o', linestyle='-', lw=3, color=colors[name], label=name, alpha=0.9)
    title_map = {'weighted_cas': 'Weighted Mean CAS', 'simple_cas': 'Simple Mean CAS', 'mcs': 'Mean MCS', 'balanced': 'Balanced Score (CAS & MCS)'}
    if ARGS.model_type == 'structural': title_map['balanced'] = 'Balanced Score (CAS, MCS & Silhouette)'
    elif ARGS.model_type == 'silhouette': title_map['balanced'] = 'Silhouette Score'
    ax.set_title(f"Bayesian Optimization Convergence\nTarget: {title_map.get(target_metric, target_metric)}", fontsize=font_size, fontweight='bold'); ax.set_xlabel('Call Number (Experiment Iteration)', fontsize=font_size, fontweight='bold'); ax.set_ylabel(f"Best Score Found", fontsize=font_size, fontweight='bold')
    legend = ax.legend(title='Strategy', fontsize=font_size, loc='best', title_fontsize=font_size); legend.get_title().set_fontweight('bold'); [text.set_fontweight('bold') for text in legend.get_texts()]
    ax.tick_params(axis='both', which='major', labelsize=font_size, width=1.2, direction='out', length=6); [label.set_fontweight('bold') for label in ax.get_xticklabels() + ax.get_yticklabels()]; ax.set_xlim(left=0, right=max_x + 1 if max_x > 0 else 1)
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_optimizer_convergence.png"); plt.savefig(output_path, dpi=500, bbox_inches='tight'); plt.close()
    print(f"✅ Success! Saved convergence plot to: {output_path}")
def plot_exact_scores_per_trial(results, target_metric, output_dir, output_prefix):
    """(Stage 1) Generates a plot of the exact score for each trial with publication-quality styling."""
    print("\n--- Generating per-trial exact score plot with publication-quality style ---")
    plt.style.use('seaborn-v0_8-white'); fig, ax = plt.subplots(figsize=(22, 10)); ax.grid(False)
    colors, font_size, max_x = {'Exploit': '#d62728', 'BO-EI': "#fcbe06", 'Explore': "#9015d2"}, 28, 0
    for name, result in results.items():
        if name in colors:
            exact_scores = -np.array(result.func_vals)
            x = np.arange(1, len(exact_scores) + 1); max_x = max(max_x, x.max())
            ax.plot(x, exact_scores, marker='.', linestyle='-', lw=2.5, color=colors[name], label=name, alpha=0.85)

    title_map = {'weighted_cas': 'Weighted Mean CAS', 'simple_cas': 'Simple Mean CAS', 'mcs': 'Mean MCS', 'balanced': 'Balanced Score (CAS & MCS)'}
    if ARGS and ARGS.model_type == 'structural': title_map['balanced'] = 'Balanced Score (CAS, MCS & Silhouette)'
    elif ARGS and ARGS.model_type == 'silhouette': title_map['balanced'] = 'Silhouette Score'
    ax.set_title(f"Per-Trial Score Progression\nTarget: {title_map.get(target_metric, target_metric)}", fontsize=font_size, fontweight='bold')
    ax.set_xlabel('Call Number (Experiment Iteration)', fontsize=font_size, fontweight='bold'); ax.set_ylabel(f"Score of Individual Trial", fontsize=font_size, fontweight='bold')
    legend = ax.legend(title='Strategy', fontsize=font_size, loc='best', title_fontsize=font_size); legend.get_title().set_fontweight('bold'); [text.set_fontweight('bold') for text in legend.get_texts()]
    ax.tick_params(axis='both', which='major', labelsize=font_size, width=1.2, direction='out', length=6); [label.set_fontweight('bold') for label in ax.get_xticklabels() + ax.get_yticklabels()]; ax.set_xlim(left=0, right=max_x + 1 if max_x > 0 else 1)
    
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_per_trial_exact_scores.png")
    plt.savefig(output_path, dpi=500, bbox_inches='tight'); plt.close()
    print(f"✅ Success! Saved per-trial exact score plot to: {output_path}")
def _get_metric_and_strategy_from_filename(filename):
    """(Stage 1) Helper to parse metric and strategy from a .skopt filename."""
    base = os.path.basename(filename).lower()
    if 'bo_ei' in base: strategy = 'BO-EI'
    elif 'exploit' in base: strategy = 'Exploit'
    elif 'explore' in base: strategy = 'Explore'
    else: strategy = (m.group(1).capitalize() if (m := re.search(r'_(\w+)_opt_result', base)) else 'Unknown')
    if 'weighted_cas' in base: metric_label = 'Weighted CAS (%)'
    elif 'simple_cas' in base: metric_label = 'Simple CAS (%)'
    elif 'mcs' in base: metric_label = 'Mean MCS (%)'
    elif 'balanced' in base and ARGS.model_type == 'silhouette': metric_label = 'Silhouette Score'
    elif 'balanced' in base: metric_label = 'Balanced Score (%)'
    else: metric_label = 'Objective Score (%)'
    return metric_label, strategy
def _style_skopt_axes(fig):
    """(Stage 1) Applies bold styling to all axes in a matplotlib figure."""
    for ax in fig.get_axes():
        ax.xaxis.label.set_fontweight('bold'); ax.yaxis.label.set_fontweight('bold')
        for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontweight('bold')
def generate_skopt_visualizations(skopt_files, output_prefix_base, target_metric):
    """(Stage 1) Loads saved .skopt results and generates detailed visualizations for each strategy."""
    print("\n--- Generating detailed skopt visualizations (Evaluations & Objective Landscape) ---")
    for f in skopt_files:
        try:
            res = load(f)
            metric_label, strategy = _get_metric_and_strategy_from_filename(f)
            clean_title = f"{strategy} ({metric_label.replace(' (%)', '')})"
            print(f"  -> Processing plots for strategy: {strategy}")
            plot_evaluations(res); fig_eval = plt.gcf(); fig_eval.set_size_inches(14, 14); fig_eval.suptitle(f'Pairwise Parameter Evaluations: {clean_title}', fontsize=16, y=0.98, fontweight='bold'); _style_skopt_axes(fig_eval); fig_eval.text(0.5, -0.02, "Diagonal: Distribution of tested values. Off-diagonal: Correlation between parameters.", ha='center', va='top', fontsize=10, fontweight='bold'); fig_eval.savefig(f"{output_prefix_base}_{strategy}_evaluations.png", dpi=300, bbox_inches='tight'); plt.close(fig_eval)
            plot_objective(res, n_points=50); fig_obj = plt.gcf(); fig_obj.set_size_inches(14, 14); fig_obj.suptitle(f'Objective Function Landscape: {clean_title}', fontsize=16, y=0.98, fontweight='bold')
            for ax in fig_obj.get_axes():
                if "Partial dependence" in ax.get_ylabel():
                    ax.set_ylabel(metric_label, fontweight='bold')
                    yticks = ax.get_yticks()
                    if np.any(yticks < 0):
                        if '%' in metric_label:
                            ax.set_yticklabels([f"{-y:.1f}" for y in yticks])
                        else: # For non-percentage scores like silhouette
                            ax.set_yticklabels([f"{-y:.2f}" for y in yticks])
            _style_skopt_axes(fig_obj); fig_obj.text(0.5, -0.02, f"Diagonal: Effect of a single parameter on the score.\nOff-diagonal: Interaction effects between two parameters.", ha='center', va='top', fontsize=10, fontweight='bold'); fig_obj.savefig(f"{output_prefix_base}_{strategy}_objective.png", dpi=300, bbox_inches='tight'); plt.close(fig_obj)
            print(f"     ✅ Saved Evaluations and Objective plots for {strategy}")
        except Exception as e: print(f"  [ERROR] Could not generate skopt plots for {f}. Reason: {e}")

# ==============================================================================
# ==============================================================================
# --- *** STAGE 2 & HELPER FUNCTIONS *** ---
# ==============================================================================
# ==============================================================================

# ==============================================================================
# --- START: INTEGRATED REFINEMENT JOURNEY SUMMARY FUNCTION ---
# ==============================================================================
def summarize_annotation_journey(input_file, output_file):
    """
    (Refinement Helper) Reads a detailed annotation scores log and creates a 
    summarized, wide-format table tracking each cell type across analysis stages.

    Args:
        input_file (str): Path to the input CSV file 
                          (e.g., 'sc_analysis_repro_combined_cluster_annotation_scores.csv').
        output_file (str): Path to save the summarized output CSV file.
    """
    try:
        print(f"Reading input file for journey summary: {input_file}")
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        sys.exit(1)

    # --- 1. Data Cleaning and Pre-aggregation ---
    # MODIFICATION START: Standardize column names for aggregation
    df.rename(columns={
        'Total_Cells_in_Group': 'Total_Cells',
        'Matching_Individual_Predictions': 'Matching_Cells',
        'Cluster_Annotation_Score_CAS (%)': 'CAS_Score'
    }, inplace=True)
    # MODIFICATION END
    
    df['source_level'] = df['source_level'].replace('initial_high_confidence', 'initial')

    print("Aggregating data for each analysis stage...")
    agg_df = df.groupby(['Consensus_Cell_Type', 'source_level']).agg(
        Total_Cells=('Total_Cells', 'sum'),
        Matching_Cells=('Matching_Cells', 'sum')
    ).reset_index()

    agg_df['Aggregated_CAS_%'] = (agg_df['Matching_Cells'] / agg_df['Total_Cells']) * 100

    # --- 2. Pivoting Data to Wide Format ---
    print("Pivoting data to create summary table...")
    pivot_df = agg_df.pivot(
        index='Consensus_Cell_Type',
        columns='source_level',
        values=['Total_Cells', 'Matching_Cells', 'Aggregated_CAS_%']
    )

    # --- 3. Formatting the Output DataFrame ---
    def sort_key(level):
        if level == 'initial':
            return 0
        match = re.search(r'refinement_depth_(\d+)', level)
        return int(match.group(1)) if match else 999

    all_stages = sorted(df['source_level'].unique(), key=sort_key)
    
    final_columns = ['Consensus_Cell_Type']
    column_mapping = {
        'Total_Cells': 'Total_Cells',
        'Matching_Cells': 'Matching_Predictions',
        'Aggregated_CAS_%': 'CAS_Score_(%)'
    }
    
    pivot_df.columns = [f'{col[1]}_{column_mapping[col[0]]}' for col in pivot_df.columns]

    for stage in all_stages:
        for suffix in column_mapping.values():
            final_columns.append(f'{stage}_{suffix}')
            
    summary_df = pivot_df.reset_index().reindex(columns=final_columns)

    for col in summary_df.columns:
        if 'CAS_Score' in col:
            summary_df[col] = summary_df[col].map('{:.2f}'.format).replace('nan', '-')
        elif 'Cells' in col or 'Predictions' in col:
            summary_df[col] = summary_df[col].apply(lambda x: int(x) if pd.notna(x) else '-')

    summary_df.fillna('-', inplace=True)
    
    # --- 4. Save to CSV ---
    print(f"✅ Success! Saving cell type journey summary report to: {output_file}")
    summary_df.to_csv(output_file, index=False)

# ==============================================================================
# --- END: INTEGRATED REFINEMENT JOURNEY SUMMARY FUNCTION ---
# ==============================================================================

def _generate_greyed_out_umap_plot(adata, cas_df, threshold, cas_aggregation_method, output_path, title, legend_fontsize=8):
    """
    (Stage 2 Helper) Generates a UMAP plot highlighting low-confidence cells in grey.
    This function is used for the *initial* Stage 2 run to identify the first batch of failing cells.
    """
    print(f"--- Identifying cells from clusters with CAS < {threshold}% for initial plot ---")
    failing_clusters_df = cas_df[cas_df['Cluster_Annotation_Score_CAS (%)'] < threshold]

    if failing_clusters_df.empty:
        print("✅ No clusters found below the threshold. All cells will be colored by type.")
        failing_cell_indices = []
    else:
        if cas_aggregation_method == 'leiden':
            if 'Cluster_ID (Leiden)' not in failing_clusters_df.columns:
                print(f"[ERROR] Column 'Cluster_ID (Leiden)' not in CAS file for greyed-out plot. Skipping.")
                return
            failing_ids = failing_clusters_df['Cluster_ID (Leiden)'].astype(str).tolist()
            failing_cell_indices = adata.obs[adata.obs['leiden'].isin(failing_ids)].index
        elif cas_aggregation_method == 'consensus':
            if 'Consensus_Cell_Type' not in failing_clusters_df.columns:
                 print(f"[ERROR] Column 'Consensus_Cell_Type' not in CAS file for greyed-out plot. Skipping.")
                 return
            failing_ids = failing_clusters_df['Consensus_Cell_Type'].tolist()
            failing_cell_indices = adata.obs[adata.obs['ctpt_consensus_prediction'].isin(failing_ids)].index
        else:
            print(f"[ERROR] Invalid CAS aggregation method provided to greyed-out plot function: {cas_aggregation_method}")
            return
        print(f"       -> Found {len(failing_ids)} failing groups: {failing_ids}")
        print(f"       -> Total cells identified as low-confidence for plotting: {len(failing_cell_indices)}")

    # Create a temporary annotation column for plotting
    plot_annotation_col = 'plot_annotation_greyed'
    adata.obs[plot_annotation_col] = adata.obs['ctpt_consensus_prediction'].astype(str)
    
    if len(failing_cell_indices) > 0:
        low_conf_label = 'Low-Confidence (<{:.0f}%)'.format(threshold)
        adata.obs.loc[failing_cell_indices, plot_annotation_col] = low_conf_label
    
    adata.obs[plot_annotation_col] = adata.obs[plot_annotation_col].astype('category')

    # Create a custom color palette to ensure consistency and add grey
    original_cats = adata.obs['ctpt_consensus_prediction'].cat.categories.tolist()
    # Use a consistent, large palette
    palette_to_use = sc.pl.palettes.godsnot_102 if len(original_cats) > 28 else sc.pl.palettes.default_102
    color_map = {cat: color for cat, color in zip(original_cats, palette_to_use)}
    
    if len(failing_cell_indices) > 0:
        color_map[low_conf_label] = '#bbbbbb'  # Medium grey

    # Generate the plot in memory
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(
            adata,
            color=plot_annotation_col,
            palette=color_map,
            title=title,
            legend_loc='right margin',
            legend_fontsize=legend_fontsize,
            frameon=False,
            size=10,
            show=False,
            save=False
        )
    
    _bold_right_margin_legend(output_path)
    plt.close()

    # Clean up the temporary column from adata.obs
    del adata.obs[plot_annotation_col]

def _bold_right_margin_legend(fig_path):
    """(Stage 2) Finds legend in current figure, makes text bold, and saves."""
    fig = plt.gcf()
    for ax in fig.axes:
        if (leg := ax.get_legend()) is not None:
            for txt in leg.get_texts(): txt.set_fontweight('bold')
    fig.savefig(fig_path, dpi=plt.rcParams['savefig.dpi'], bbox_inches='tight')
def reformat_dotplot_data(fraction_df: pd.DataFrame, top_genes_df: pd.DataFrame, output_dir: str, output_prefix: str, groupby_key: str):
    """(Stage 2) Reformats dot plot fraction data to a gene-centric sparse table."""
    print(f"[INFO] Reformatting dot plot data for '{groupby_key}'...")
    cell_types = top_genes_df['group'].unique().tolist()
    output_rows = []
    for _, row in top_genes_df.iterrows():
        gene, group = row['names'], row['group']
        fraction = fraction_df.loc[group, gene]
        new_row_data = {'Gene': gene, **{ct: '' for ct in cell_types}}
        new_row_data[group] = fraction
        output_rows.append(new_row_data)
    reformatted_df = pd.DataFrame(output_rows)[['Gene'] + cell_types]
    reformatted_csv_path = os.path.join(output_dir, f"{output_prefix}_dotplot_fractions_{groupby_key}_reformatted.csv")
    reformatted_df.to_csv(reformatted_csv_path, index=False)
    print(f"       -> Saved reformatted fraction data to: {reformatted_csv_path}")
def extract_fraction_data_and_calculate_mcs(adata: anndata.AnnData, output_dir: str, output_prefix: str, groupby_key: str, top_genes_df: pd.DataFrame, cli_args):
    """(Stage 2, Single-Sample) Calculates and saves expression fractions and the MCS."""
    print(f"[INFO] Calculating MCS and expression fractions for '{groupby_key}'...")
    if groupby_key not in adata.obs.columns: print(f"[ERROR] Grouping key '{groupby_key}' not found. Skipping MCS."); return None
    unique_top_genes = top_genes_df['names'].unique().tolist()
    data_df = sc.get.obs_df(adata, keys=[groupby_key] + unique_top_genes, use_raw=(adata.raw is not None))
    fraction_df = data_df.groupby(groupby_key).apply(lambda x: (x[unique_top_genes] > 0).mean())
    mcs_scores = {cell_type: fraction_df.loc[cell_type, top_genes_df[top_genes_df['group'] == cell_type]['names']].mean() for cell_type in top_genes_df['group'].unique()}
    mcs_df = pd.DataFrame.from_dict(mcs_scores, orient='index', columns=['MCS']); mcs_df.index.name = 'Cell_Type'
    mcs_df.to_csv(os.path.join(output_dir, f"{output_prefix}_marker_concordance_scores.csv")); print(f"       -> Saved MCS scores.")
    fraction_df.to_csv(os.path.join(output_dir, f"{output_prefix}_dotplot_fractions_{groupby_key}.csv")); print(f"       -> Saved full fraction data.")
    reformat_dotplot_data(fraction_df, top_genes_df, output_dir, output_prefix, groupby_key)
    return mcs_df
def extract_fraction_data_for_dotplot(adata: anndata.AnnData, output_dir: str, output_prefix: str, groupby_key: str, top_genes_df: pd.DataFrame):
    """(Stage 2, Multi-Sample) Calculates and saves expression fractions for dotplot."""
    print(f"[INFO] Calculating expression fractions for dotplot for '{groupby_key}'...")
    if groupby_key not in adata.obs.columns:
        print(f"[ERROR] Grouping key '{groupby_key}' not found in adata.obs. Skipping.")
        return
    unique_top_genes = top_genes_df['names'].unique().tolist()
    data_df = sc.get.obs_df(adata, keys=[groupby_key] + unique_top_genes, use_raw=(adata.raw is not None))
    fraction_df = data_df.groupby(groupby_key).apply(lambda x: (x[unique_top_genes] > 0).mean())
    output_csv_path = os.path.join(output_dir, f"{output_prefix}_dotplot_fractions_{groupby_key}.csv")
    fraction_df.to_csv(output_csv_path)
    print(f"       -> Saved full fraction data to: {output_csv_path}")
    reformat_dotplot_data(fraction_df, top_genes_df, output_dir, output_prefix, groupby_key)

def run_stage_two_final_analysis(cli_args, optimal_params, output_dir, data_dir=None, adata_input=None):
    """
    (Stage 2) Executes the detailed single-sample analysis pipeline using
    parameters discovered in Stage 1. All outputs are saved to a subdirectory.
    Can either load data from `data_dir` or use a pre-loaded `adata_input`.
    """
    print("--- Initializing Stage 2: CAS-MCS Scoring Pipeline with Optimal Parameters ---")

    random.seed(cli_args.seed); np.random.seed(cli_args.seed); sc.settings.njobs = 1
    print(f"[INFO] Global random seed set to: {cli_args.seed}")

    sc.settings.verbosity = 3; sc.logging.print_header()
    sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False, dpi_save=cli_args.fig_dpi)
    sc.settings.figdir = output_dir
    print(f"[INFO] Scanpy version: {sc.__version__}")
    print(f"[INFO] Outputting to subdirectory: {os.path.abspath(output_dir)}")

    print("\n--- Step 1: Loading Data ---")
    if adata_input is not None:
        print("       -> Using provided AnnData object for analysis.")
        adata = adata_input.copy()
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
    elif data_dir is not None:
        print(f"       -> Loading from directory: {data_dir}")
        adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
        adata.var_names_make_unique(); adata.layers["counts"] = adata.X.copy()
    else:
        raise ValueError("Must provide 'data_dir' or 'adata_input' to run_stage_two_final_analysis.")
    print(f"       -> Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    print("\n--- Step 2: Quality Control and Filtering ---")
    adata.var['mt'] = adata.var_names.str.contains(MITO_REGEX_PATTERN, regex=True)
    print(f"       -> Identified {adata.var['mt'].sum()} mitochondrial genes using robust regex.")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=False)
    plt.savefig(os.path.join(output_dir, f"{cli_args.final_run_prefix}_qc_plots_before_filtering.png")); plt.close()

    sc.pp.filter_cells(adata, min_genes=cli_args.min_genes)
    sc.pp.filter_cells(adata, max_genes=cli_args.max_genes)
    adata = adata[adata.obs.pct_counts_mt < cli_args.max_pct_mt, :]
    sc.pp.filter_genes(adata, min_cells=cli_args.min_cells)
    print(f"       -> Filtered dims: {adata.n_obs} cells, {adata.n_vars} genes")

    print("\n--- Step 3: Normalization, HVG, Scaling (using optimal params) ---")
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata); adata.raw = adata.copy()

    if all(p is not None for p in [cli_args.hvg_min_mean, cli_args.hvg_max_mean, cli_args.hvg_min_disp]):
        print("[INFO] Using two-step sequential HVG selection.")
        sc.pp.highly_variable_genes(adata, min_mean=cli_args.hvg_min_mean, max_mean=cli_args.hvg_max_mean, min_disp=cli_args.hvg_min_disp)
        hvg_df = adata.var[adata.var.highly_variable].sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:optimal_params['n_hvg']]
        adata.var['highly_variable'] = False; adata.var.loc[top_genes, 'highly_variable'] = True
    else:
        print(f"[INFO] Using rank-based HVG selection with n_top_genes={optimal_params['n_hvg']}")
        sc.pp.highly_variable_genes(adata, n_top_genes=optimal_params['n_hvg'], flavor='seurat_v3')

    sc.pl.highly_variable_genes(adata, save=f"_{cli_args.final_run_prefix}_hvg_plot.png", show=False); plt.close()
    adata = adata[:, adata.var.highly_variable]
    print(f"       -> Final selection: {adata.n_vars} highly variable genes for downstream analysis.")
    sc.pp.scale(adata, max_value=10)

    print("\n--- Step 4: Dimensionality Reduction and Clustering (using optimal params) ---")
    # --- BUG FIX START ---
    # Robustly cap the number of PCs by both cells and genes, crucial for refinement runs.
    n_pcs_to_compute = min(cli_args.n_pcs_compute, adata.n_obs - 1, adata.n_vars - 1)
    # --- BUG FIX END ---
    n_pcs_to_use = min(optimal_params['n_pcs'], n_pcs_to_compute)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs_to_compute, random_state=cli_args.seed)
    sc.pl.pca_variance_ratio(adata, log=True, n_pcs=n_pcs_to_compute, save=f"_{cli_args.final_run_prefix}_pca_variance.png", show=False); plt.close()

    sc.pp.neighbors(adata, n_neighbors=optimal_params['n_neighbors'], n_pcs=n_pcs_to_use, random_state=cli_args.seed)
    sc.tl.leiden(adata, resolution=optimal_params['resolution'], random_state=cli_args.seed)
    sc.tl.umap(adata, random_state=cli_args.seed)
    silhouette_avg = silhouette_score(adata.obsm['X_pca'][:, :n_pcs_to_use], adata.obs['leiden'])
    print(f"       -> Average Silhouette Score for Leiden clustering: {silhouette_avg:.3f}")
    sc.pl.umap(adata, color='leiden', legend_fontweight='bold', legend_loc='on data', title=f'Leiden Clusters ({adata.obs["leiden"].nunique()} clusters)\nSilhouette: {silhouette_avg:.3f}', palette=sc.pl.palettes.godsnot_102, save=f"_{cli_args.final_run_prefix}_umap_leiden.png", show=False, size=10); plt.close()

    print("\n--- Step 5: CellTypist Annotation and CAS Calculation ---")
    model_ct = models.Model.load(cli_args.model_path)
    print("[INFO] Annotating cells using the full log-normalized transcriptome (from adata.raw)...")
    predictions = celltypist.annotate(adata.raw.to_adata(), model=model_ct, majority_voting=False)
    adata.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels'].astype('category')
    if 'conf_score' in predictions.predicted_labels.columns: adata.obs['ctpt_confidence'] = predictions.predicted_labels['conf_score']

    sc.pl.umap(adata, color='ctpt_individual_prediction', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Per-Cell CellTypist Annotation ({adata.obs["ctpt_individual_prediction"].nunique()} types)', show=False, size=10)
    _bold_right_margin_legend(os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_per_cell_celltypist.png")); plt.close()

    cluster2label = adata.obs.groupby('leiden')['ctpt_individual_prediction'].agg(lambda x: x.value_counts().idxmax()).to_dict()
    adata.obs['ctpt_consensus_prediction'] = adata.obs['leiden'].map(cluster2label).astype('category')
    sc.pl.umap(adata, color='ctpt_consensus_prediction', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Cluster-Consensus CellTypist Annotation ({adata.obs["ctpt_consensus_prediction"].nunique()} types)', show=False, size=10)
    _bold_right_margin_legend(os.path.join(output_dir, f"{cli_args.final_run_prefix}_cluster_celltypist_umap.png")); plt.close()
    
    leiden_purity_results = []
    leiden_groups = adata.obs.groupby('leiden')
    for leiden_id, group in leiden_groups:
        consensus_name = group['ctpt_consensus_prediction'].iloc[0]
        # MODIFICATION START: Standardize column name
        leiden_purity_results.append({
            "Cluster_ID (Leiden)": leiden_id, "Consensus_Cell_Type": consensus_name, "Total_Cells_in_Group": len(group),
            "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == consensus_name).sum(),
            "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == consensus_name).sum() / len(group) if len(group) > 0 else 0
        })
        # MODIFICATION END
    cas_leiden_df = pd.DataFrame(leiden_purity_results).sort_values(by="Cluster_Annotation_Score_CAS (%)", ascending=False)
    cas_leiden_output_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_leiden_cluster_annotation_scores.csv")
    cas_leiden_df.to_csv(cas_leiden_output_path, index=False)
    print(f"       -> Saved Leiden-based CAS (technical purity) scores to: {cas_leiden_output_path}")

    consensus_purity_results = []
    for name, group in adata.obs.groupby('ctpt_consensus_prediction'):
        # MODIFICATION START: Standardize column name
        consensus_purity_results.append({
            "Consensus_Cell_Type": name, "Total_Cells_in_Group": len(group),
            "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == name).sum(),
            "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == name).sum() / len(group) if len(group) > 0 else 0
        })
        # MODIFICATION END
    cas_consensus_df = pd.DataFrame(consensus_purity_results).sort_values(by="Cluster_Annotation_Score_CAS (%)", ascending=False)
    cas_consensus_output_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_consensus_group_annotation_scores.csv")
    cas_consensus_df.to_csv(cas_consensus_output_path, index=False)
    print(f"       -> Saved Consensus-based CAS (final label purity) scores to: {cas_consensus_output_path}")

    if cli_args.cas_aggregation_method == 'leiden':
        cas_df_for_refinement, cas_path_for_refinement = cas_leiden_df, cas_leiden_output_path
        print("[INFO] Using Leiden-based CAS report for refinement thresholding.")
    else: # 'consensus'
        cas_df_for_refinement, cas_path_for_refinement = cas_consensus_df, cas_consensus_output_path
        print("[INFO] Using Consensus-based CAS report for refinement thresholding.")
    
    if cli_args.cas_refine_threshold is not None:
        print("\n--- Generating verification UMAP with low-confidence cells highlighted ---")
        greyed_umap_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_low_confidence_greyed.png")
        _generate_greyed_out_umap_plot(adata=adata, cas_df=cas_df_for_refinement, threshold=cli_args.cas_refine_threshold, cas_aggregation_method=cli_args.cas_aggregation_method, output_path=greyed_umap_path, title=f'Consensus Annotation (Failing Cells <{cli_args.cas_refine_threshold}% CAS in Grey)', legend_fontsize=8)
        print(f"       -> Saved greyed-out UMAP to: {greyed_umap_path}")

    print("\n--- Step 6: Marker Gene Analysis and MCS Calculation ---")
    marker_groupby_key = 'ctpt_consensus_prediction'; top_genes_df, mcs_df = None, None
    if marker_groupby_key in adata.obs.columns:
        adata.obs[marker_groupby_key] = adata.obs[marker_groupby_key].cat.remove_unused_categories()
        print(f"       -> Cleaned '{marker_groupby_key}': {adata.obs[marker_groupby_key].nunique()} active categories")
    valid_labels = adata.obs[marker_groupby_key].value_counts()[lambda x: x > 1].index.tolist()
    if len(valid_labels) < 2: print(f"[WARNING] Skipping marker gene analysis: Fewer than 2 consensus groups with >1 cell.")
    else:
        sc.tl.rank_genes_groups(adata, marker_groupby_key, groups=valid_labels, method='wilcoxon', use_raw=True, key_added=f"wilcoxon_{marker_groupby_key}")
        marker_df = sc.get.rank_genes_groups_df(adata, key=f"wilcoxon_{marker_groupby_key}", group=None)

        is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
        filtered_rows = [sub[~sub['names'].map(is_mito)].head(cli_args.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
        top_genes_df = pd.concat(filtered_rows, ignore_index=True)

        # [MODIFICATION START: Wrap DotPlot in Try/Except]
        try:
            print(f"       -> Attempting to generate marker gene dotplot...")
            with plt.rc_context({'font.size': 18, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
                # Safety check for categories existence
                valid_cats = set(adata.obs[marker_groupby_key].unique())
                cats_to_plot = [c for c in top_genes_df['group'].unique().tolist() if c in valid_cats]
                
                if cats_to_plot:
                    sc.pl.dotplot(adata, 
                                  var_names=top_genes_df.groupby('group')['names'].apply(list).to_dict(), 
                                  groupby=marker_groupby_key, 
                                  categories_order=cats_to_plot, 
                                  use_raw=True, 
                                  save=f"_{cli_args.final_run_prefix}_markers_celltypist_dotplot.png", 
                                  show=False)
                    plt.close()
                else:
                    print("       -> [SKIP] No valid categories overlap for dotplot.")
        except Exception as e:
            print(f"       -> [WARNING] Dotplot generation failed. Error: {e}")
            print("       -> Pipeline continuing without this plot...")
        # [MODIFICATION END]

        mcs_df = extract_fraction_data_and_calculate_mcs(adata, output_dir, cli_args.final_run_prefix, marker_groupby_key, top_genes_df, cli_args)
        if mcs_df is not None and top_genes_df is not None:
            top_genes_agg = top_genes_df.groupby('group')['names'].apply(', '.join).reset_index().rename(columns={'names': f'Top_{cli_args.n_top_genes}_Markers', 'group': 'Cell_Type'})
            pd.merge(mcs_df, top_genes_agg, on='Cell_Type')[['Cell_Type', 'MCS', f'Top_{cli_args.n_top_genes}_Markers']].to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_mcs_and_top_markers.csv"), index=False); print(f"       -> Saved combined MCS and Top Markers.")

    print("\n--- Step 7: Optional Manual-Style Annotation & Scoring ---")
    if cli_args.cellmarker_db and os.path.exists(cli_args.cellmarker_db):
        try:
            print(f"       -> Annotating using marker DB: {cli_args.cellmarker_db}")
            
            sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', use_raw=True, key_added='wilcoxon_leiden')
            leiden_markers_df = sc.get.rank_genes_groups_df(adata, key='wilcoxon_leiden', group=None)

            header = pd.read_csv(cli_args.cellmarker_db, nrows=0).columns.tolist()
            type_col, gene_col = ('cell_name', 'Symbol') if 'cell_name' in header and 'Symbol' in header else (('Cell Type', 'Cell Marker') if 'Cell Type' in header and 'Cell Marker' in header else (None, None))
            if not type_col: raise ValueError("Marker DB must contain ('cell_name', 'Symbol') or ('Cell Type', 'Cell Marker') columns.")
            print(f"       -> Auto-detected format: TYPE='{type_col}', GENE='{gene_col}'")

            db_df = pd.read_csv(cli_args.cellmarker_db)
            db_markers_dict = defaultdict(set)
            for _, row in db_df.iterrows():
                if pd.notna(row.get(gene_col)) and pd.notna(row.get(type_col)):
                    db_markers_dict[row[type_col]].update({m.strip().upper() for m in str(row[gene_col]).split(',')})
            print(f"       -> Aggregated markers for {len(db_markers_dict)} unique cell types.")

            cluster_annotations = {}
            for cluster in adata.obs['leiden'].cat.categories:
                cluster_genes = set(leiden_markers_df[leiden_markers_df['group'] == cluster].head(cli_args.n_top_genes)['names'].str.upper())
                scores = {cell_type: len(cluster_genes.intersection(db_genes)) / (len(cluster_genes.union(db_genes)) or 1) for cell_type, db_genes in db_markers_dict.items()}
                best_cell_type = max(scores, key=scores.get) if scores else None
                cluster_annotations[cluster] = best_cell_type if best_cell_type and scores[best_cell_type] > 0 else f"Unknown_{cluster}"

            adata.obs['manual_annotation'] = adata.obs['leiden'].map(cluster_annotations).astype('category')
            pd.DataFrame.from_dict(cluster_annotations, orient='index', columns=['AssignedType']).to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_leiden_to_manual_annotation.csv"))
            sc.pl.umap(adata, color='manual_annotation', title='Manual Cluster Annotation', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, size=10, show=False)
            _bold_right_margin_legend(os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_manual_annotation.png")); plt.close()

            print("       -> Calculating Marker Capture Score for manual annotation...")
            score_results = []
            leiden_degs_structured = adata.uns['wilcoxon_leiden']['names']
            for cluster_id, assigned_label in cluster_annotations.items():
                if pd.isna(assigned_label) or assigned_label.startswith("Unknown"): continue
                reference_genes = db_markers_dict.get(assigned_label, set())
                if not reference_genes: continue
                
                cluster_degs_for_capture = {g.upper() for g in leiden_degs_structured[cluster_id][:cli_args.n_degs_for_capture]}
                captured_genes = cluster_degs_for_capture.intersection(reference_genes)
                score = (len(captured_genes) / len(reference_genes)) * 100
                
                score_results.append({
                    "Cluster_ID": cluster_id, "Assigned_Cell_Type": assigned_label, "Marker_Capture_Score (%)": score,
                    "Captured_Genes_Count": len(captured_genes), "Total_Reference_Genes": len(reference_genes),
                    "Captured_Genes_List": ", ".join(sorted(list(captured_genes)))
                })
            
            if score_results:
                capture_df = pd.DataFrame(score_results).sort_values(by="Marker_Capture_Score (%)", ascending=False)
                capture_df.to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_manual_annotation_marker_capture_scores.csv"), index=False)
                print(f"       -> Saved Marker Capture Scores.")

        except Exception as e:
            print(f"[ERROR] Manual annotation/scoring failed. Reason: {e}")
    else:
        print("[INFO] Cell marker DB not provided or not found. Skipping manual-style annotation.")

    print("\n--- Step 8: Exporting All Results ---")
    cols_to_save = [col for col in ['leiden', 'ctpt_individual_prediction', 'ctpt_confidence', 'ctpt_consensus_prediction', 'manual_annotation'] if col in adata.obs.columns]
    adata.obs[cols_to_save].to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_all_annotations.csv"))
    print(f"       -> All cell annotations saved."); adata.write(os.path.join(output_dir, f"{cli_args.final_run_prefix}_final_processed.h5ad")); print(f"       -> Final AnnData object saved.")

    print("\n--- Step 9: Verifying Metrics Against Optimization Run ---")
    total_matching = (adata.obs['ctpt_individual_prediction'].astype(str) == adata.obs['ctpt_consensus_prediction'].astype(str)).sum()
    weighted_cas = (total_matching / len(adata.obs)) * 100 if len(adata.obs) > 0 else 0.0
    
    simple_cas = 0.0
    if cli_args.cas_aggregation_method == 'leiden':
        simple_cas_groups = [(g['ctpt_individual_prediction'].astype(str) == g['ctpt_consensus_prediction'].astype(str).iloc[0]).mean() * 100 for _, g in adata.obs.groupby('leiden') if not g.empty]
        simple_cas = np.mean(simple_cas_groups) if simple_cas_groups else 0.0
    elif cli_args.cas_aggregation_method == 'consensus':
        cas_per_consensus_group = [(g['ctpt_individual_prediction'].astype(str) == g['ctpt_consensus_prediction'].astype(str).iloc[0]).mean() * 100 for _, g in adata.obs.groupby('ctpt_consensus_prediction') if not g.empty]
        simple_cas = np.mean(cas_per_consensus_group) if cas_per_consensus_group else 0.0

    mean_mcs = mcs_df['MCS'].mean() if mcs_df is not None and not mcs_df.empty else 0.0
    target_map = {'simple_cas': "Simple Mean CAS", 'weighted_cas': "Weighted Mean CAS", 'mcs': "Mean MCS", 'balanced': "Balanced Score"}
    print("\n" + "="*50 + f"\n--- Final Verification Summary (Single-Sample) ---\nOptimization Target from Stage 1: {target_map.get(cli_args.target, 'N/A')}\nRandom Seed Used: {cli_args.seed}\n\n--- Optimal Parameters Used ---\nBest n_hvg: {optimal_params['n_hvg']}\nBest n_pcs: {n_pcs_to_use}\nBest n_neighbors: {optimal_params['n_neighbors']}\nBest resolution: {optimal_params['resolution']:.3f}\n")
    if cli_args.target == 'simple_cas': print(f"Highest_simple_mean_cas_pct: {simple_cas:.2f}\nCorresponding_weighted_mean_cas_pct: {weighted_cas:.2f}\nCorresponding_mean_mcs_pct: {mean_mcs * 100:.2f}\nCorresponding_silhouette_score: {silhouette_avg:.3f}\n")
    elif cli_args.target == 'weighted_cas': print(f"Highest_weighted_mean_cas_pct: {weighted_cas:.2f}\nCorresponding_simple_mean_cas_pct: {simple_cas:.2f}\nCorresponding_mean_mcs_pct: {mean_mcs * 100:.2f}\nCorresponding_silhouette_score: {silhouette_avg:.3f}\n")
    elif cli_args.target in ['mcs', 'balanced']: print(f"Target Score ({cli_args.target}): {mean_mcs * 100 if cli_args.target=='mcs' else 'N/A'}\nCorresponding_weighted_mean_cas_pct: {weighted_cas:.2f}\nCorresponding_simple_mean_cas_pct: {simple_cas:.2f}\nCorresponding_mean_mcs_pct: {mean_mcs * 100:.2f}\nCorresponding_silhouette_score: {silhouette_avg:.3f}\n")
    print(f"Final_n_individual_labels: {adata.obs['ctpt_individual_prediction'].nunique()}\nFinal_n_consensus_labels: {adata.obs['ctpt_consensus_prediction'].nunique()}\n" + "="*50)
    
    return adata, cas_path_for_refinement

def run_stage_two_final_analysis_multi_sample(cli_args, optimal_params, output_dir, wt_path=None, treated_path=None, adata_input=None):
    """
    (Stage 2) Executes the detailed two-sample integration analysis pipeline using
    parameters discovered in Stage 1.
    """
    print("--- Initializing Stage 2: Two-Sample Integration Pipeline with Optimal Parameters ---")

    random.seed(cli_args.seed); np.random.seed(cli_args.seed); sc.settings.njobs = 1
    print(f"[INFO] Global random seed set to: {cli_args.seed} for reproducibility.")

    CONDITION_OF_INTEREST, REFERENCE_CONDITION = 'Treated', 'WT'
    FINAL_ANNOTATION_COLUMN = 'ctpt_consensus_prediction'

    sc.settings.verbosity = 3; sc.logging.print_header()
    sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False, dpi_save=cli_args.fig_dpi)
    sc.settings.figdir = output_dir
    print(f"[INFO] Scanpy version: {sc.__version__}\n[INFO] Outputting to subdirectory: {os.path.abspath(output_dir)}")

    if adata_input is not None:
        print("\n--- Step 1 & 2: Using Provided AnnData for Analysis ---")
        adata = adata_input.copy()
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        if 'sample' not in adata.obs.columns:
            raise ValueError("Input AnnData for multi-sample refinement must contain a 'sample' column in .obs")
    elif wt_path is not None and treated_path is not None:
        SAMPLE_INFO = {'WT': {'path': wt_path}, 'Treated': {'path': treated_path}}
        print("\n--- Step 1 & 2: Loading and Concatenating Datasets ---")
        adatas = {sid: sc.read_10x_mtx(info['path'], var_names='gene_symbols', cache=True) for sid, info in SAMPLE_INFO.items()}
        for sid, adata_sample in adatas.items():
            adata_sample.var_names_make_unique(); adata_sample.obs['sample'] = sid
        adata = anndata.AnnData.concatenate(*adatas.values(), batch_key='sample', batch_categories=list(adatas.keys()))
    else:
        raise ValueError("Must provide ('wt_path', 'treated_path') or 'adata_input' to run_stage_two_final_analysis_multi_sample.")

    print("\n--- Step 3: Quality Control ---")
    adata.var['mt'] = adata.var_names.str.contains(MITO_REGEX_PATTERN, regex=True)
    print(f"       -> Identified {adata.var['mt'].sum()} mitochondrial genes using robust regex.")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None, log1p=False)

    sc.pp.filter_cells(adata, min_genes=cli_args.min_genes)
    sc.pp.filter_cells(adata, max_genes=cli_args.max_genes)
    adata = adata[adata.obs.pct_counts_mt < cli_args.max_pct_mt, :]
    sc.pp.filter_genes(adata, min_cells=cli_args.min_cells)
    print(f"       -> Filtered dims: {adata.n_obs} cells, {adata.n_vars} genes")

    print("\n--- Step 4: Normalization, HVG selection, Scaling ---")
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata); adata.raw = adata.copy()

    if all(p is not None for p in [cli_args.hvg_min_mean, cli_args.hvg_max_mean, cli_args.hvg_min_disp]):
        print("[INFO] Using two-step sequential HVG selection.")
        sc.pp.highly_variable_genes(adata, min_mean=cli_args.hvg_min_mean, max_mean=cli_args.hvg_max_mean, min_disp=cli_args.hvg_min_disp, batch_key='sample')
        hvg_df = adata.var[adata.var.highly_variable].sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:optimal_params['n_hvg']]
        adata.var['highly_variable'] = False; adata.var.loc[top_genes, 'highly_variable'] = True
    else:
        print(f"[INFO] Using rank-based HVG selection with n_top_genes={optimal_params['n_hvg']}")
        sc.pp.highly_variable_genes(adata, n_top_genes=optimal_params['n_hvg'], flavor='seurat_v3', batch_key='sample')

    sc.pl.highly_variable_genes(adata, save=f"_{cli_args.final_run_prefix}_hvg_plot.png", show=False); plt.close()
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"       -> Final selection: {adata.n_vars} highly variable genes for downstream analysis.")
    sc.pp.scale(adata, max_value=10)

    print("\n--- Step 5: PCA and Batch Correction with Harmony ---")
    # --- BUG FIX START ---
    # Robustly cap the number of PCs by both cells and genes, crucial for refinement runs.
    n_pcs_to_compute = min(cli_args.n_pcs_compute, adata.n_obs - 1, adata.n_vars - 1)
    # --- BUG FIX END ---
    n_pcs_to_use = min(optimal_params['n_pcs'], n_pcs_to_compute)
    print(f"[INFO] Computing {n_pcs_to_compute} PCs, using top {n_pcs_to_use} for downstream.")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs_to_compute, random_state=cli_args.seed)

    try:
        import harmonypy as hm
        print("harmonypy is installed. Performing batch correction.")
        sc.external.pp.harmony_integrate(adata, key='sample', basis='X_pca', adjusted_basis='X_pca_harmony', random_state=cli_args.seed)
        pca_rep_key = 'X_pca_harmony'
    except ImportError:
        print("[WARNING] harmonypy not found. Skipping Harmony integration."); pca_rep_key = 'X_pca'

    print("\n--- Step 6: Neighborhood, Clustering, and UMAP on Integrated Data ---")
    sc.pp.neighbors(adata, n_neighbors=optimal_params['n_neighbors'], n_pcs=n_pcs_to_use, use_rep=pca_rep_key, random_state=cli_args.seed)
    sc.tl.leiden(adata, resolution=optimal_params['resolution'], random_state=cli_args.seed)
    sc.tl.umap(adata, random_state=cli_args.seed)
    sc.pl.umap(adata, color='sample', title='UMAP by Sample', save=f"_{cli_args.final_run_prefix}_umap_sample.png", show=False, size=10); plt.close()

    silhouette_avg = 0.0
    try:
        if adata.obs['leiden'].nunique() > 1:
            silhouette_avg = silhouette_score(adata.obsm[pca_rep_key][:, :n_pcs_to_use], adata.obs['leiden'])
            print(f"       -> Average Silhouette Score for Leiden clustering (on '{pca_rep_key}'): {silhouette_avg:.3f}")
        else:
            print("       -> Silhouette score not computed (only 1 cluster).")
    except Exception as e:
        print(f"       -> [WARNING] Could not compute silhouette score: {e}")

    sc.pl.umap(adata, color='leiden', legend_loc='on data', legend_fontweight='bold', title=f'Leiden Clusters (res={optimal_params["resolution"]})\nSilhouette: {silhouette_avg:.3f}', palette=sc.pl.palettes.godsnot_102, save=f"_{cli_args.final_run_prefix}_umap_leiden.png", show=False, size=10); plt.close()


    print("\n--- Step 7: Cell Type Annotation with CellTypist ---")
    top_genes_df = None
    cas_path_for_refinement = None # Initialize path
    
    if cli_args.model_path and os.path.exists(cli_args.model_path):
        model_ct = models.Model.load(cli_args.model_path)
        print("[INFO] Annotating cells using the full log-normalized transcriptome (from adata.raw)...")
        predictions = celltypist.annotate(adata.raw.to_adata(), model=model_ct, majority_voting=False)
        adata.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels']
        
        # START: ADDED PLOT FOR PER-CELL ANNOTATION
        sc.pl.umap(adata, color='ctpt_individual_prediction', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Per-Cell CellTypist Annotation ({adata.obs["ctpt_individual_prediction"].nunique()} types)', show=False, size=10)
        _bold_right_margin_legend(os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_per_cell_celltypist.png")); plt.close()
        # END: ADDED PLOT

        adata.obs[FINAL_ANNOTATION_COLUMN] = adata.obs.groupby('leiden')['ctpt_individual_prediction'].transform(lambda x: x.value_counts().idxmax()).astype('category')
        adata.obs[FINAL_ANNOTATION_COLUMN] = adata.obs[FINAL_ANNOTATION_COLUMN].cat.remove_unused_categories()
        print(f"       -> Cleaned '{FINAL_ANNOTATION_COLUMN}': {adata.obs[FINAL_ANNOTATION_COLUMN].nunique()} active categories")
        leiden_purity_results = []
        leiden_groups = adata.obs.groupby('leiden')
        for leiden_id, group in leiden_groups:
            consensus_name = group[FINAL_ANNOTATION_COLUMN].iloc[0]
            # MODIFICATION START: Standardize column name
            leiden_purity_results.append({
                "Cluster_ID (Leiden)": leiden_id, "Consensus_Cell_Type": consensus_name, "Total_Cells_in_Group": len(group),
                "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == consensus_name).sum(),
                "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == consensus_name).sum() / len(group) if len(group) > 0 else 0
            })
            # MODIFICATION END
        cas_leiden_df = pd.DataFrame(leiden_purity_results).sort_values(by="Cluster_Annotation_Score_CAS (%)", ascending=False)
        cas_leiden_output_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_leiden_cluster_annotation_scores.csv")
        cas_leiden_df.to_csv(cas_leiden_output_path, index=False)
        print(f"       -> Saved Leiden-based CAS (technical purity) scores to: {cas_leiden_output_path}")

        consensus_purity_results = []
        for name, group in adata.obs.groupby(FINAL_ANNOTATION_COLUMN):
            # MODIFICATION START: Standardize column name
            consensus_purity_results.append({
                "Consensus_Cell_Type": name, "Total_Cells_in_Group": len(group),
                "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == name).sum(),
                "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == name).sum() / len(group) if len(group) > 0 else 0
            })
            # MODIFICATION END
        cas_consensus_df = pd.DataFrame(consensus_purity_results).sort_values(by="Cluster_Annotation_Score_CAS (%)", ascending=False)
        cas_consensus_output_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_consensus_group_annotation_scores.csv")
        cas_consensus_df.to_csv(cas_consensus_output_path, index=False)
        print(f"       -> Saved Consensus-based CAS (final label purity) scores to: {cas_consensus_output_path}")

        if cli_args.cas_aggregation_method == 'leiden':
            cas_df_for_refinement, cas_path_for_refinement = cas_leiden_df, cas_leiden_output_path
            print("[INFO] Using Leiden-based CAS report for refinement thresholding.")
        else: # 'consensus'
            cas_df_for_refinement, cas_path_for_refinement = cas_consensus_df, cas_consensus_output_path
            print("[INFO] Using Consensus-based CAS report for refinement thresholding.")

        sc.pl.umap(adata, color=FINAL_ANNOTATION_COLUMN, title='Cluster-Consensus CellTypist Annotation', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, size=10, show=False)
        fig_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_cluster_celltypist_umap.png"); _bold_right_margin_legend(fig_path); plt.close()

        if cli_args.cas_refine_threshold is not None:
            print("\n--- Generating verification UMAP with low-confidence cells highlighted ---")
            greyed_umap_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_low_confidence_greyed.png")
            _generate_greyed_out_umap_plot(adata=adata, cas_df=cas_df_for_refinement, threshold=cli_args.cas_refine_threshold, cas_aggregation_method=cli_args.cas_aggregation_method, output_path=greyed_umap_path, title=f'Consensus Annotation (Failing Cells <{cli_args.cas_refine_threshold}% CAS in Grey)', legend_fontsize=8)
            print(f"       -> Saved greyed-out UMAP to: {greyed_umap_path}")

        marker_key = f"wilcoxon_{FINAL_ANNOTATION_COLUMN}"

        # [NEW SAFETY CHECK START]
        # Ensure we have at least 2 groups with data before ranking genes
        unique_groups = adata.obs[FINAL_ANNOTATION_COLUMN].dropna().unique()
        if len(unique_groups) < 2:
            print(f"       -> [SKIP] Skipping marker analysis/dotplot. Only {len(unique_groups)} cell type(s) present in this subset.")
        else:
            # [EXISTING CODE MOVED INSIDE ELSE BLOCK]
            sc.tl.rank_genes_groups(adata, FINAL_ANNOTATION_COLUMN, method='wilcoxon', use_raw=True, key_added=marker_key)
            marker_df = sc.get.rank_genes_groups_df(adata, key=marker_key, group=None)
            is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
            filtered_rows = [sub[~sub['names'].map(is_mito)].head(cli_args.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            top_genes_df = pd.concat(filtered_rows, ignore_index=True)

            # [MODIFICATION START: Ultra-Safe Try/Except Block]
            try:
                print(f"       -> Attempting to generate marker gene dotplot...")
                with plt.rc_context({'font.size': 18, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
                    genes_to_plot = top_genes_df.groupby('group')['names'].apply(list).to_dict()
                    
                    # Explicitly verify categories exist in .obs before plotting
                    valid_cats_in_obs = set(adata.obs[FINAL_ANNOTATION_COLUMN].unique())
                    safe_categories_order = [cat for cat in list(genes_to_plot.keys()) if cat in valid_cats_in_obs]
                    
                    if safe_categories_order:
                        sc.pl.dotplot(adata, var_names=genes_to_plot, groupby=FINAL_ANNOTATION_COLUMN, 
                                      categories_order=safe_categories_order, 
                                      use_raw=True, save=f"_{cli_args.final_run_prefix}_markers_celltypist_dotplot.png", show=False)
                        plt.close()
                    else:
                         print("       -> [SKIP] No valid categories found for dotplot ordering.")
            except Exception as e:
                print(f"       -> [WARNING] Dotplot generation failed. Error: {e}")
                print("       -> Pipeline continuing without this plot...")
            # [MODIFICATION END]
            
            extract_fraction_data_for_dotplot(adata, output_dir, cli_args.final_run_prefix, FINAL_ANNOTATION_COLUMN, top_genes_df)
        # [NEW SAFETY CHECK END]
    else:
        print("[INFO] CellTypist not run. Using Leiden clusters for downstream analysis.")
        adata.obs[FINAL_ANNOTATION_COLUMN] = adata.obs['leiden'].astype('category')

    print("\n--- Step 8: Find Marker Genes for raw Leiden clusters ---")
    if 'leiden' in adata.obs.columns:
        adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
        print(f"       -> Cleaned 'leiden': {adata.obs['leiden'].nunique()} active categories")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', use_raw=True, key_added='wilcoxon_leiden')
    sc.pl.rank_genes_groups(adata, n_genes=20, key='wilcoxon_leiden', sharey=False, save=f"_{cli_args.final_run_prefix}_markers_leiden.png", show=False); plt.close()

    print("\n--- Step 9: Manual Annotation ---")
    if cli_args.cellmarker_db and os.path.exists(cli_args.cellmarker_db):
        try:
            print(f"       -> Annotating using marker DB: {cli_args.cellmarker_db}")
            header = pd.read_csv(cli_args.cellmarker_db, nrows=0).columns.tolist()
            type_col, gene_col = ('cell_name', 'Symbol') if 'cell_name' in header and 'Symbol' in header else (('Cell Type', 'Cell Marker') if 'Cell Type' in header and 'Cell Marker' in header else (None, None))
            if not type_col: raise ValueError("Marker DB must contain ('cell_name', 'Symbol') or ('Cell Type', 'Cell Marker') columns.")
            print(f"       -> Auto-detected format: TYPE='{type_col}', GENE='{gene_col}'")
            db_df = pd.read_csv(cli_args.cellmarker_db)
            db_markers_dict = defaultdict(set)
            for _, row in db_df.iterrows():
                if pd.notna(row.get(gene_col)) and pd.notna(row.get(type_col)):
                    db_markers_dict[row[type_col]].update({m.strip().upper() for m in str(row[gene_col]).split(',')})
            print(f"       -> Aggregated markers for {len(db_markers_dict)} unique cell types.")
            leiden_markers_df = sc.get.rank_genes_groups_df(adata, key='wilcoxon_leiden', group=None)
            cluster_annotations = {}
            for cluster in adata.obs['leiden'].cat.categories:
                cluster_genes = set(leiden_markers_df[leiden_markers_df['group'] == cluster].head(cli_args.n_top_genes)['names'].str.upper())
                scores = {cell_type: len(cluster_genes.intersection(db_genes)) / (len(cluster_genes.union(db_genes)) or 1) for cell_type, db_genes in db_markers_dict.items()}
                best_cell_type = max(scores, key=scores.get) if scores else None
                cluster_annotations[cluster] = best_cell_type if best_cell_type and scores[best_cell_type] > 0 else f"Unknown_{cluster}"
            adata.obs['manual_annotation'] = adata.obs['leiden'].map(cluster_annotations).astype('category')
            pd.DataFrame.from_dict(cluster_annotations, orient='index', columns=['AssignedType']).to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_leiden_to_manual_annotation.csv"))
            sc.pl.umap(adata, color='manual_annotation', title='Manual Cluster Annotation', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, size=10, show=False)
            fig_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_manual_annotation.png"); _bold_right_margin_legend(fig_path); plt.close()
            
            print("       -> Calculating Marker Capture Score for manual annotation...")
            score_results = []
            leiden_degs_structured = adata.uns['wilcoxon_leiden']['names']
            for cluster_id, assigned_label in cluster_annotations.items():
                if pd.isna(assigned_label) or assigned_label.startswith("Unknown"): continue
                reference_genes = db_markers_dict.get(assigned_label, set())
                if not reference_genes: continue
                cluster_degs_for_capture = {g.upper() for g in leiden_degs_structured[cluster_id][:cli_args.n_degs_for_capture]}
                captured_genes = cluster_degs_for_capture.intersection(reference_genes)
                score = (len(captured_genes) / len(reference_genes)) * 100
                score_results.append({"Cluster_ID": cluster_id, "Assigned_Cell_Type": assigned_label, "Marker_Capture_Score (%)": score, "Captured_Genes_Count": len(captured_genes), "Total_Reference_Genes": len(reference_genes), "Captured_Genes_List": ", ".join(sorted(list(captured_genes)))})
            if score_results:
                capture_df = pd.DataFrame(score_results).sort_values(by="Marker_Capture_Score (%)", ascending=False)
                capture_df.to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_manual_annotation_marker_capture_scores.csv"), index=False)
                print(f"       -> Saved Marker Capture Scores.")
        except Exception as e: print(f"[ERROR] Manual annotation failed. Reason: {e}")
    else: print("[INFO] Cell marker DB not provided or not found. Skipping manual annotation.")

    print("\n--- Step 10: Compositional Analysis ---")
    composition_counts = pd.crosstab(adata.obs[FINAL_ANNOTATION_COLUMN], adata.obs['sample'])
    composition_perc = composition_counts.div(composition_counts.sum(axis=0), axis=1) * 100
    composition_perc.to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_composition_percentages.csv"))
    fig, ax = plt.subplots(figsize=(12, 8)); composition_perc.T.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_ylabel('Percentage of Cells'); ax.set_xlabel('Sample'); ax.set_title('Cell Type Composition by Sample')
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{cli_args.final_run_prefix}_composition_barchart.png")); plt.close()

    print("\n--- Step 11: Differential Gene Expression (DGE) Analysis ---")
    dge_results = []
    for cell_type in adata.obs[FINAL_ANNOTATION_COLUMN].cat.categories:
        print(f"  -> Running DGE for: {cell_type}")
        sub_adata = adata[(adata.obs[FINAL_ANNOTATION_COLUMN] == cell_type)].copy()
        
        # MODIFICATION START: More robust check for DGE viability
        counts_per_sample = sub_adata.obs['sample'].value_counts()
        if (CONDITION_OF_INTEREST not in counts_per_sample) or \
           (REFERENCE_CONDITION not in counts_per_sample) or \
           (counts_per_sample[CONDITION_OF_INTEREST] < 2) or \
           (counts_per_sample[REFERENCE_CONDITION] < 2):
            print(f"     [SKIP] Not enough cells for DGE in '{cell_type}'. "
                  f"Counts: {counts_per_sample.to_dict()}. Need at least 2 cells in both '{CONDITION_OF_INTEREST}' and '{REFERENCE_CONDITION}'.")
            continue
        # MODIFICATION END

        try:
            sc.tl.rank_genes_groups(sub_adata, 'sample', groups=[CONDITION_OF_INTEREST], reference=REFERENCE_CONDITION, method='wilcoxon', use_raw=True, key_added='dge_result')
            dge_df = sc.get.rank_genes_groups_df(sub_adata, key='dge_result', group=CONDITION_OF_INTEREST); dge_df['cell_type'] = cell_type
            dge_results.append(dge_df)
        except Exception as e: print(f"     [ERROR] DGE failed for '{cell_type}'. Reason: {e}")
    if dge_results:
        pd.concat(dge_results, ignore_index=True).to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_DGE_Treated_vs_WT_by_celltype.csv"), index=False)
        print("DGE analysis complete. Full results saved.")
    else: print("No DGE results were generated.")

    print("\n--- Step 12: Final Marker Heatmap ---")
    marker_key = f"wilcoxon_{FINAL_ANNOTATION_COLUMN}"
    if marker_key in adata.uns and top_genes_df is not None and not top_genes_df.empty:
        genes_to_plot_list = top_genes_df['names'].unique().tolist()
        print(f"       -> Generating heatmap with top {cli_args.n_top_genes} non-mitochondrial marker genes per cell type.")
        
        # --- FIX: Explicitly re-calculate the dendrogram for the current adata state ---
        sc.tl.dendrogram(adata, groupby=FINAL_ANNOTATION_COLUMN)
        # --- END FIX ---
        
        sc.pl.heatmap(adata, var_names=genes_to_plot_list, groupby=FINAL_ANNOTATION_COLUMN, show=False, dendrogram=True, save=f"_{cli_args.final_run_prefix}_top_markers_heatmap.png"); plt.close()
    else: 
        print(f"[WARNING] Marker key '{marker_key}' not found or markers not computed. Cannot generate heatmap.")

    print("\n--- Step 13: Saving Final AnnData Object ---")
    adata.write(os.path.join(output_dir, f"{cli_args.final_run_prefix}_final_processed.h5ad"))
    print(f"       -> Final annotated AnnData object saved.")

    print("\n" + "="*50 + f"\n--- Final Parameters Summary (Multi-Sample) ---\nRandom Seed Used: {cli_args.seed}\n\n--- Optimal Parameters Used ---\nBest n_hvg: {optimal_params['n_hvg']}\nBest n_pcs: {n_pcs_to_use}\nBest n_neighbors: {optimal_params['n_neighbors']}\nBest resolution: {optimal_params['resolution']:.3f}\n")
    print(f"Final_n_leiden_clusters: {adata.obs['leiden'].nunique()}")
    print(f"Final_silhouette_score: {silhouette_avg:.3f}")
    if FINAL_ANNOTATION_COLUMN in adata.obs.columns: print(f"Final_n_consensus_labels: {adata.obs[FINAL_ANNOTATION_COLUMN].nunique()}")
    print("="*50 + "\n\n--- MULTI-SAMPLE ANALYSIS PIPELINE COMPLETE ---")
    
    return adata, cas_path_for_refinement

# ==============================================================================
# ==============================================================================
# --- *** STAGE 3 & 4: REFINEMENT PIPELINE *** ---
# ==============================================================================
# ==============================================================================

# =========================================================================================
# === NEW HELPER FUNCTION for cumulative UMAP plotting during refinement ===
# =========================================================================================
def _generate_cumulative_refinement_umap(adata_full, failing_cell_indices, threshold, output_path, title, legend_fontsize=8):
    """
    (Refinement Helper) Generates a UMAP plot showing the CUMULATIVE progress of refinement.
    
    This function operates on the main, full AnnData object. It colors cells by their
    most up-to-date 'combined_annotation' and specifically colors a provided list of
    'failing_cell_indices' in grey.

    Args:
        adata_full (anndata.AnnData): The complete, original AnnData object from Stage 2.
                                      Must contain 'X_umap' and a 'combined_annotation' column.
        failing_cell_indices (pd.Index): Cell indices of the low-confidence cells for this level.
        threshold (float): The CAS percentage threshold, used for the plot title.
        output_path (str): Full path to save the output PNG image.
        title (str): The title for the UMAP plot.
        legend_fontsize (int): The font size for the legend text.
    """
    print(f"--- Generating cumulative refinement UMAP showing {len(failing_cell_indices)} failing cells in grey ---")

    # Use a copy of the full AnnData object to avoid modifying it
    adata_plot = adata_full.copy()

    # Create a temporary annotation column for plotting
    plot_annotation_col = 'plot_annotation_cumulative'
    adata_plot.obs[plot_annotation_col] = adata_plot.obs['combined_annotation'].astype(str)
    
    low_conf_label = 'Low-Confidence (<{:.0f}%)'.format(threshold)
    if len(failing_cell_indices) > 0:
        # Mark the currently failing cells with the special grey label
        adata_plot.obs.loc[failing_cell_indices, plot_annotation_col] = low_conf_label
    
    adata_plot.obs[plot_annotation_col] = adata_plot.obs[plot_annotation_col].astype('category')

    # Create a custom color palette that includes all seen annotations plus grey
    all_seen_cats = adata_plot.obs['combined_annotation'].cat.categories.tolist()
    # Use a consistent, large palette
    palette_to_use = sc.pl.palettes.godsnot_102
    color_map = {cat: color for cat, color in zip(all_seen_cats, palette_to_use)}
    
    if len(failing_cell_indices) > 0:
        color_map[low_conf_label] = '#bbbbbb'  # Medium grey

    # Generate the plot in memory
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(
            adata_plot,
            color=plot_annotation_col,
            palette=color_map,
            title=title,
            legend_loc='right margin',
            legend_fontsize=legend_fontsize,
            frameon=False,
            size=10,
            show=False,
            save=False
        )
    
    _bold_right_margin_legend(output_path)
    plt.close()
    print(f"       -> Saved cumulative progress UMAP to: {output_path}")

# =========================================================================================
# === NEW/REPLACEMENT FUNCTION for multi-level refinement ===
# =========================================================================================
def run_iterative_refinement_pipeline(args, adata_s2, cas_csv_path_s2):
    """
    Orchestrates iterative refinement and produces cumulative UMAPs showing the
    results of each refinement step, saving all final results.

    For each refinement level, this function:
    1. Identifies failing cells from the previous analysis level.
    2. Runs Stage 1 (BO) and Stage 2 (Final Analysis) on these failing cells. This
       preserves the detailed analysis files for each subset.
    3. Updates a master annotation column ('combined_annotation') in the main AnnData object.
    4. Generates a CUMULATIVE UMAP showing the *result* of this refinement:
       - Newly passing cells are now colored with their new labels.
       - Cells that *still* fail are shown in grey.
    5. Repeats this process up to `args.refinement_depth`.
    6. Saves a final, combined AnnData object and CSV with the new refined annotations.
    """
    global adata_base, model, RANDOM_SEED, ARGS, CURRENT_OPTIMIZATION_TARGET, CURRENT_STRATEGY_NAME, TRIAL_METADATA

    print("\n\n" + "="*80 + "\n### STARTING STAGE 3/4: ITERATIVE REFINEMENT PIPELINE ###\n" + "="*80)
    
    # --- Step 1: Initial Setup ---
    main_stage1_dir = os.path.join(args.output_dir, "stage_1_bayesian_optimization")
    stage2_output_dir = os.path.join(args.output_dir, "stage_2_final_analysis")
    
    # These variables track the state from one loop to the next
    current_cas_csv_path = cas_csv_path_s2
    adata_to_check = adata_s2 # AnnData from the previous analysis level
    adata_raw_full = adata_s2.raw.to_adata() # Full, original raw data
    
    all_refinement_cas_paths = []
    # This 'combined_annotation' is the master column that gets progressively updated
    adata_s2.obs['combined_annotation'] = adata_s2.obs['ctpt_consensus_prediction'].astype('category')

    original_bo_output_dir = args.output_dir
    original_bo_prefix = args.output_prefix
    original_final_run_prefix = args.final_run_prefix
    
    for depth in range(1, args.refinement_depth + 1):
        print("\n\n" + "#"*70 + f"\n### REFINEMENT DEPTH {depth}/{args.refinement_depth} ###\n" + "#"*70)

        # --- Step 2: Identify failing cells from the PREVIOUS level ---
        if not os.path.exists(current_cas_csv_path):
             print(f"[ERROR] Cannot find CAS file for depth {depth-1} at: {current_cas_csv_path}. Stopping refinement.")
             break

        cas_df_prev_level = pd.read_csv(current_cas_csv_path)

        # Identify the cells that are the INPUT for this refinement round
        if args.cas_aggregation_method == 'leiden':
            failing_cluster_ids = cas_df_prev_level[cas_df_prev_level['Cluster_Annotation_Score_CAS (%)'] < args.cas_refine_threshold]['Cluster_ID (Leiden)'].astype(str).tolist()
            if not failing_cluster_ids:
                print(f"✅ All clusters at depth {depth-1} met the {args.cas_refine_threshold}% CAS threshold. No further refinement needed.")
                break
            print(f"Found {len(failing_cluster_ids)} Leiden clusters below threshold at depth {depth-1}: {failing_cluster_ids}")
            failing_cell_indices_input = adata_to_check.obs[adata_to_check.obs['leiden'].isin(failing_cluster_ids)].index
        
        else: # 'consensus'
            failing_clusters = cas_df_prev_level[cas_df_prev_level['Cluster_Annotation_Score_CAS (%)'] < args.cas_refine_threshold]['Consensus_Cell_Type'].tolist()
            if not failing_clusters:
                print(f"✅ All clusters at depth {depth-1} met the {args.cas_refine_threshold}% CAS threshold. No further refinement needed.")
                break
            print(f"Found {len(failing_clusters)} consensus types below threshold at depth {depth-1}: {', '.join(failing_clusters)}")
            failing_cell_indices_input = adata_to_check.obs[adata_to_check.obs['ctpt_consensus_prediction'].isin(failing_clusters)].index

        if len(failing_cell_indices_input) < args.min_cells_refinement:
            print(f"\n[STOP] Stopping refinement. Only {len(failing_cell_indices_input)} failing cells, below the minimum of {args.min_cells_refinement}.\n")
            break

        # Isolate the subset of raw data for this refinement level
        adata_refine_raw = adata_raw_full[failing_cell_indices_input, :].copy()
        print(f"Isolated {adata_refine_raw.n_obs} cells for refinement analysis at depth {depth}.")

        # --- ADDED CHECK: Verify Gene Content (Pre-Flight Check) ---
        # Create a temp object to check if we have enough HVGs to satisfy the optimizer
        check_adata = adata_refine_raw.copy()
        sc.pp.normalize_total(check_adata, target_sum=1e4)
        sc.pp.log1p(check_adata)

        n_found_hvgs = 0
        # Check if using strict thresholding (Two-Step)
        if all(p is not None for p in [args.hvg_min_mean, args.hvg_max_mean, args.hvg_min_disp]):
             sc.pp.highly_variable_genes(
                check_adata,
                min_mean=args.hvg_min_mean,
                max_mean=args.hvg_max_mean,
                min_disp=args.hvg_min_disp,
                batch_key='sample' if 'sample' in check_adata.obs.columns else None
            )
             n_found_hvgs = check_adata.var.highly_variable.sum()
        else:
            # If rank based, the limit is just the total number of genes available
            n_found_hvgs = check_adata.n_vars

        MIN_HVG_LIMIT = 200 # As defined in Integer(200, 20000)
        if n_found_hvgs < MIN_HVG_LIMIT:
            print(f"\n[STOP] Stopping refinement at depth {depth}.")
            print(f"       Reason: Found only {n_found_hvgs} potential HVGs (or total genes), which is less than the optimizer lower bound of {MIN_HVG_LIMIT}.")
            print(f"       This usually happens when the subset is too homogeneous or too small.")
            break
        # -----------------------------------------------------------
        
        # --- Step 3: Run Stage 1 (BO) on the subset ---
        stage1_refinement_dir = os.path.join(main_stage1_dir, f"refinement_depth_{depth}")
        os.makedirs(stage1_refinement_dir, exist_ok=True)
        
        args.output_dir = stage1_refinement_dir
        args.output_prefix = f"{original_bo_prefix}_refinement_depth_{depth}"
        
        print(f"\n--- [Depth {depth}] Running new Bayesian Optimization on subset. Outputs will be in: {stage1_refinement_dir} ---")
        refinement_bo_results = run_stage_one_optimization(args, adata_input=adata_refine_raw)
        refinement_optimal_params = refinement_bo_results['params']
        
        args.output_dir = original_bo_output_dir # Restore for next loop or finalization
        args.output_prefix = original_bo_prefix

        # --- Step 4: Run Stage 2 (Final Analysis) on the subset ---
        stage2_refinement_dir = os.path.join(stage2_output_dir, f"refinement_depth_{depth}")
        os.makedirs(stage2_refinement_dir, exist_ok=True)

        print(f"\n--- [Depth {depth}] Running Final Analysis on subset. Outputs will be in: {stage2_refinement_dir} ---")
        is_multi_sample_refinement = 'sample' in adata_refine_raw.obs.columns
        
        args.final_run_prefix = f"{original_final_run_prefix}_refinement_depth_{depth}"

        if is_multi_sample_refinement:
            adata_refinement_processed, cas_csv_path_refinement = run_stage_two_final_analysis_multi_sample(
                cli_args=args, optimal_params=refinement_optimal_params, output_dir=stage2_refinement_dir, adata_input=adata_refine_raw
            )
        else:
            adata_refinement_processed, cas_csv_path_refinement = run_stage_two_final_analysis(
                cli_args=args, optimal_params=refinement_optimal_params, output_dir=stage2_refinement_dir, adata_input=adata_refine_raw
            )
        
        args.final_run_prefix = original_final_run_prefix # Restore for next loop

        # --- Step 5: Update master annotation in the main adata_s2 object ---
        all_refinement_cas_paths.append(cas_csv_path_refinement)
        
        refinement_annotations = adata_refinement_processed.obs['ctpt_consensus_prediction']
        
        # Ensure new categories from this refinement are added to the master annotation column
        current_categories = adata_s2.obs['combined_annotation'].cat.categories.tolist()
        new_labels = refinement_annotations.unique()
        new_categories_to_add = [label for label in new_labels if label not in current_categories]
        if new_categories_to_add:
            print(f"       -> Adding new categories to master list: {new_categories_to_add}")
            adata_s2.obs['combined_annotation'] = adata_s2.obs['combined_annotation'].cat.add_categories(new_categories_to_add)

        # Now, perform the assignment of new labels
        adata_s2.obs.loc[refinement_annotations.index, 'combined_annotation'] = refinement_annotations.astype(str)
        adata_s2.obs['combined_annotation'] = adata_s2.obs['combined_annotation'].astype('category') # Recategorize
        adata_s2.obs['combined_annotation'] = adata_s2.obs['combined_annotation'].cat.remove_unused_categories()
        print(f"--- [Depth {depth}] Updated {len(refinement_annotations)} cell annotations in the main object. Active categories: {adata_s2.obs['combined_annotation'].nunique()} ---")
        print(f"--- [Depth {depth}] Updated {len(refinement_annotations)} cell annotations in the main object. ---")
        
        # --- Step 6: Identify cells that ARE STILL FAILING for the cumulative plot ---
        failing_cell_indices_output = pd.Index([]) # Default to empty
        if os.path.exists(cas_csv_path_refinement):
            cas_df_this_level = pd.read_csv(cas_csv_path_refinement)
            
            if args.cas_aggregation_method == 'leiden':
                still_failing_ids = cas_df_this_level[cas_df_this_level['Cluster_Annotation_Score_CAS (%)'] < args.cas_refine_threshold]['Cluster_ID (Leiden)'].astype(str).tolist()
                if still_failing_ids:
                    failing_cell_indices_output = adata_refinement_processed.obs[adata_refinement_processed.obs['leiden'].isin(still_failing_ids)].index
            else: # 'consensus'
                still_failing_types = cas_df_this_level[cas_df_this_level['Cluster_Annotation_Score_CAS (%)'] < args.cas_refine_threshold]['Consensus_Cell_Type'].tolist()
                if still_failing_types:
                    failing_cell_indices_output = adata_refinement_processed.obs[adata_refinement_processed.obs['ctpt_consensus_prediction'].isin(still_failing_types)].index
        
        print(f"--- [Depth {depth}] Found {len(failing_cell_indices_output)} cells that are *still* failing. These will be grey in the cumulative UMAP. ---")

        # --- Step 7: Generate the cumulative UMAP showing the *result* of this depth's analysis ---
        # This plot is saved in the current depth's directory and shows progress.
        greyed_umap_path = os.path.join(stage2_refinement_dir, f"{args.final_run_prefix}_refinement_depth_{depth}_umap_cumulative_result.png")
        _generate_cumulative_refinement_umap(
            adata_full=adata_s2, # Use the main object with updated 'combined_annotation'
            failing_cell_indices=failing_cell_indices_output, # Grey out only the cells that are still failing
            threshold=args.cas_refine_threshold,
            output_path=greyed_umap_path,
            title=f'Refinement Level {depth}: Cumulative Result\n({len(failing_cell_indices_output)} cells remain low-confidence)',
            legend_fontsize=8
        )

        # --- Step 8: Update state for the next iteration ---
        current_cas_csv_path = cas_csv_path_refinement
        adata_to_check = adata_refinement_processed

    # --- FINAL COMBINATION (after loop) ---
    print("\n\n" + "="*80 + "\n### FINALIZING ALL REFINEMENT RESULTS ###\n" + "="*80)
    
    print("--- Generating final combined UMAP plot with consistent styling ---")
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(adata_s2, color='combined_annotation', palette=sc.pl.palettes.godsnot_102, 
                legend_loc='right margin', legend_fontsize=8, 
                title='Final Annotation (High-Confidence + All Refined Levels)', 
                show=False, size=10)

    combined_umap_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_umap_combined_annotation_final.png")
    _bold_right_margin_legend(combined_umap_path); plt.close()
    print(f"✅ Success! Saved final combined UMAP to: {combined_umap_path}")

    print("--- Generating final combined CAS score sheet ---")
    passing_cas_df = pd.read_csv(cas_csv_path_s2)
    passing_cas_df = passing_cas_df[passing_cas_df['Cluster_Annotation_Score_CAS (%)'] >= args.cas_refine_threshold]
    passing_cas_df['source_level'] = 'initial_high_confidence'
    all_cas_dfs = [passing_cas_df]
    
    for i, path in enumerate(all_refinement_cas_paths):
        try:
            refinement_cas_df = pd.read_csv(path)
            refinement_cas_df['source_level'] = f'refinement_depth_{i+1}'
            all_cas_dfs.append(refinement_cas_df)
        except FileNotFoundError:
            print(f"[WARNING] Could not find CAS file for refinement level {i+1} at '{path}'. Skipping.")
    
    if len(all_cas_dfs) > 1: # Only save if refinement actually happened
        combined_cas_df = pd.concat(all_cas_dfs, ignore_index=True)
        combined_cas_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_combined_cluster_annotation_scores.csv")
        combined_cas_df.to_csv(combined_cas_path, index=False)
        print(f"✅ Success! Saved combined CAS scores to: {combined_cas_path}")

        # --- START: CALL TO THE NEW JOURNEY SUMMARY FUNCTION ---
        print("--- Generating cell type journey summary report ---")
        journey_summary_output_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_cell_type_journey_summary.csv")
        summarize_annotation_journey(
            input_file=combined_cas_path,
            output_file=journey_summary_output_path
        )
        # --- END: CALL TO THE NEW JOURNEY SUMMARY FUNCTION ---

    print("--- Saving final AnnData object and annotations CSV with refinement results ---")
    final_adata_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_final_processed_with_refinement.h5ad")
    adata_s2.write(final_adata_path)
    print(f"✅ Success! Saved final AnnData object to: {final_adata_path}")
    
    # Save the final annotations to a new CSV, including the new 'combined_annotation' column
    final_csv_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_all_annotations_with_refinement.csv")
    cols_to_save = [c for c in adata_s2.obs.columns if c in ['leiden', 'ctpt_individual_prediction', 'ctpt_confidence', 'ctpt_consensus_prediction', 'manual_annotation', 'combined_annotation']]
    adata_s2.obs[cols_to_save].to_csv(final_csv_path)
    print(f"✅ Success! Saved final annotations with refinement column to: {final_csv_path}")

# ==============================================================================
# ==============================================================================
# --- *** PIPELINE ORCHESTRATION *** ---
# ==============================================================================
# ==============================================================================

def run_stage_one_optimization(args, adata_input=None):
    """
    Executes the entire Bayesian optimization pipeline (Stage 1).
    Can load data from disk (if `adata_input` is None) or use a provided
    AnnData object (for refinement runs).
    Returns a dictionary with the best parameters.
    """
    global adata_base, model, RANDOM_SEED, ARGS, CURRENT_OPTIMIZATION_TARGET, CURRENT_STRATEGY_NAME, TRIAL_METADATA

    ARGS = args
    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    sc.settings.verbosity = 0
    sc.logging.print_header()
    os.makedirs(args.output_dir, exist_ok=True)

    if adata_input is None:
        if args.data_dir:
            print("--- Running in SINGLE-SAMPLE Mode ---")
            adata = sc.read_10x_mtx(args.data_dir, var_names='gene_symbols', cache=True)
            adata.var_names_make_unique()
            adata_merged = adata
        elif args.multi_sample:
            print("--- Running in MULTI-SAMPLE (Harmony Integration) Mode ---")
            wt_dir, treated_dir = args.multi_sample
            adatas = {'WT': sc.read_10x_mtx(wt_dir, var_names='gene_symbols', cache=True), 'Treated': sc.read_10x_mtx(treated_dir, var_names='gene_symbols', cache=True)}
            for sample_id, adata_sample in adatas.items():
                adata_sample.var_names_make_unique(); adata_sample.obs['sample'] = sample_id
            adata_merged = anndata.AnnData.concatenate(*adatas.values(), batch_key='sample', batch_categories=list(adatas.keys()))
            print(f"Combined data: {adata_merged.n_obs} cells, {adata_merged.n_vars} genes")
        else:
            raise ValueError("Invalid arguments. Must provide --data_dir or --multi_sample for the initial run.")
    else:
        print("--- Using provided AnnData object for optimization ---")
        adata_merged = adata_input.copy()

    print("\n--- Performing initial QC and normalization ---")
    if 'mt' not in adata_merged.var.columns:
        adata_merged.var['mt'] = adata_merged.var_names.str.contains(MITO_REGEX_PATTERN, regex=True)
        sc.pp.calculate_qc_metrics(adata_merged, qc_vars=['mt'], inplace=True)
        sc.pp.filter_cells(adata_merged, min_genes=args.min_genes)
        sc.pp.filter_cells(adata_merged, max_genes=args.max_genes)
        adata_merged = adata_merged[adata_merged.obs.pct_counts_mt < args.max_pct_mt, :]
        sc.pp.filter_genes(adata_merged, min_cells=args.min_cells)
    
    print(f"Data for this BO run: {adata_merged.n_obs} cells, {adata_merged.n_vars} genes")
    sc.pp.normalize_total(adata_merged, target_sum=1e4); sc.pp.log1p(adata_merged); adata_merged.raw = adata_merged.copy()
    adata_base = adata_merged.copy(); model = models.Model.load(args.model_path)
    print("Initial setup complete. Base AnnData object created for optimization.")

    local_search_space = [dim for dim in search_space]
    if all(p is not None for p in [args.hvg_min_mean, args.hvg_max_mean, args.hvg_min_disp]):
        print("\n--- Two-step HVG mode enabled. Pre-calculating gene filter... ---")
        adata_temp = adata_base.copy()
        sc.pp.highly_variable_genes(adata_temp, min_mean=args.hvg_min_mean, max_mean=args.hvg_max_mean, min_disp=args.hvg_min_disp, batch_key='sample' if 'sample' in adata_base.obs.columns else None)
        n_filtered_genes = adata_temp.var.highly_variable.sum()
        print(f"       -> Found {n_filtered_genes} genes passing thresholds.")
        original_min_hvg = next(dim.low for dim in search_space if dim.name == 'n_hvg')
        if n_filtered_genes < original_min_hvg: print(f"[ERROR] HVG filtering resulted in only {n_filtered_genes} genes, below the minimum search bound of {original_min_hvg}. Please relax your HVG filtering thresholds."); exit(1)
        for i, dim in enumerate(local_search_space):
            if dim.name == 'n_hvg':
                print(f"       -> Adjusting 'n_hvg' search space to [{original_min_hvg}, {n_filtered_genes}].")
                local_search_space[i] = Integer(original_min_hvg, n_filtered_genes, name='n_hvg'); break
    else: print("\n--- Using standard rank-based HVG selection mode. ---")

    param_names = ['n_hvg', 'n_pcs', 'n_neighbors', 'resolution']
    targets_to_run = ['balanced'] if args.target == 'all' else [args.target]
    best_params_for_stage2 = None

    for target in targets_to_run:
        target_name_map = {'weighted_cas': 'WEIGHTED MEAN CAS', 'simple_cas': 'SIMPLE MEAN CAS', 'mcs': 'MEAN MCS', 'balanced': 'BALANCED SCORE (CAS & MCS)'}
        if args.model_type == 'structural': target_name_map['balanced'] = 'BALANCED SCORE (CAS, MCS & SILHOUETTE)'
        elif args.model_type == 'silhouette': target_name_map['balanced'] = 'SILHOUETTE SCORE'
        print("\n\n" + "#"*70 + f"\n### STAGE: OPTIMIZING FOR {target_name_map[target]} ###\n" + "#"*70)
        CURRENT_OPTIMIZATION_TARGET = target
        strategies = {"Exploit": {'acq_func': 'PI', 'xi': 0.01}, "BO-EI": {'acq_func': 'EI', 'xi': 0.01}, "Explore": {'acq_func': 'EI', 'xi': 0.1}}
        output_prefix_model = f"{args.output_prefix}_{args.model_type}"
        results, skopt_file_paths = {}, []
        for name, params in strategies.items():
            print(f"\n--- Running Strategy: {name} ---"); CURRENT_STRATEGY_NAME = name; TRIAL_METADATA.clear()
            result = gp_minimize(func=objective_function, dimensions=local_search_space, n_calls=args.n_calls, random_state=RANDOM_SEED, **params)
            result.trial_metadata = list(TRIAL_METADATA); results[name] = result
            result_path = os.path.join(args.output_dir, f"{output_prefix_model}_{target}_{name.lower().replace('-','_')}_opt_result.skopt")
            dump(result, result_path, store_objective=False); skopt_file_paths.append(result_path); print(f"Saved {name} optimization state to {result_path}")

        generate_yield_csv(results, target, args.output_dir, output_prefix_model)
        plot_optimizer_paths_tsne(results, target, args.output_dir, output_prefix_model, n_points_to_show=args.n_calls)
        plot_optimizer_paths_umap(results, target, args.output_dir, output_prefix_model, n_points_to_show=args.n_calls)
        plot_optimizer_convergence(results, target, args.output_dir, output_prefix_model)
        plot_exact_scores_per_trial(results, target, args.output_dir, output_prefix_model)
        generate_skopt_visualizations(skopt_files=skopt_file_paths, output_prefix_base=os.path.join(args.output_dir, f"{output_prefix_model}_{target}"), target_metric=target)

        best_overall_score, best_result_obj, winning_strategy_name = float('inf'), None, ""
        for name, result in results.items():
            if result.fun < best_overall_score: best_overall_score, best_result_obj, winning_strategy_name = result.fun, result, name
        
        best_score_print = -best_overall_score
        format_str = ".3f" if args.model_type == 'silhouette' else ".2f"
        print(f"\n--- Analysis Complete for {target_name_map[target]} ---\nOverall best score ({best_score_print:{format_str}}) was found by the '{winning_strategy_name}' strategy.")

        best_params_for_stage2 = dict(zip(param_names, best_result_obj.x))
        final_metrics, adata_final = evaluate_final_metrics(best_params_for_stage2)
        print_final_report(target, best_params_for_stage2, final_metrics, winning_strategy_name)
        txt_path = os.path.join(args.output_dir, f"{output_prefix_model}_{target}_FINAL_best_params.txt")
        h5ad_path = os.path.join(args.output_dir, f"{output_prefix_model}_{target}_FINAL_annotated.h5ad")
        save_results_to_file(txt_path, target, best_params_for_stage2, final_metrics, winning_strategy_name)
        adata_final.write(h5ad_path)
        print(f"\nFinal optimized results for {target} saved to:\n  - {txt_path}\n  - {h5ad_path}")

    print("\n\n--- Stage 1 (Optimization) Complete ---")
    return_data = {"params": best_params_for_stage2}
    if args.data_dir:
        return_data["data_dir"] = args.data_dir
    elif args.multi_sample:
        return_data["wt_dir"], return_data["treated_dir"] = args.multi_sample
    
    return return_data


def main(parsed_args):
    """Main orchestrator for the two-stage pipeline."""
    adata_s2, cas_csv_path_s2 = None, None

    # --- STAGE 1 ---
    print("="*80 + "\n### STARTING STAGE 1: BAYESIAN PARAMETER OPTIMIZATION ###\n" + "="*80)
    stage1_output_dir = os.path.join(parsed_args.output_dir, "stage_1_bayesian_optimization")
    
    original_output_dir = parsed_args.output_dir
    parsed_args.output_dir = stage1_output_dir
    
    optimization_results = run_stage_one_optimization(parsed_args, adata_input=None)
    optimal_params = optimization_results.get("params")
    
    parsed_args.output_dir = original_output_dir 

    # --- STAGE 2 (Conditional) ---
    if optimal_params:
        print("\n\n" + "="*80 + "\n### STARTING STAGE 2: FINAL ANALYSIS ###\n" + "="*80)
        stage2_output_dir = os.path.join(parsed_args.output_dir, "stage_2_final_analysis")
        os.makedirs(stage2_output_dir, exist_ok=True)
        print(f"Stage 2 outputs will be saved to: {os.path.abspath(stage2_output_dir)}")
        
        if parsed_args.data_dir:
            adata_s2, cas_csv_path_s2 = run_stage_two_final_analysis(
                cli_args=parsed_args, optimal_params=optimal_params, output_dir=stage2_output_dir, data_dir=parsed_args.data_dir
            )
        elif parsed_args.multi_sample:
            wt_path, treated_path = parsed_args.multi_sample
            adata_s2, cas_csv_path_s2 = run_stage_two_final_analysis_multi_sample(
                cli_args=parsed_args, optimal_params=optimal_params, output_dir=stage2_output_dir, wt_path=wt_path, treated_path=treated_path
            )
    else:
        print("\n\n" + "="*80 + "\n### SKIPPING STAGE 2 ###\nStage 1 did not complete successfully.\n" + "="*80)
        print("\n--- Integrated pipeline finished with errors. ---")
        return

    # --- STAGE 3 & 4 (OPTIONAL REFINEMENT) ---
    if parsed_args.cas_refine_threshold is not None:
        if adata_s2 is not None and cas_csv_path_s2 is not None and os.path.exists(cas_csv_path_s2):
            run_iterative_refinement_pipeline(
                args=parsed_args, adata_s2=adata_s2, cas_csv_path_s2=cas_csv_path_s2
            )
        else:
            print(f"[WARNING] --cas_refine_threshold was set, but Stage 2 did not produce the necessary outputs to proceed. Skipping refinement.")

    print("\n--- Integrated pipeline finished successfully! ---")