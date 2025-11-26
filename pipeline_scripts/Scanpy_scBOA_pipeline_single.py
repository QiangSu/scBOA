#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive command-line pipeline for single-cell analysis and quantitative
annotation scoring (CAS & MCS).

This script performs an end-to-end analysis from 10x Genomics data, including
QC, clustering, and multiple annotation strategies. It calculates two key metrics:
- Cluster Annotation Score (CAS): The purity of CellTypist annotations within a
  consensus cluster.
- Marker Concordance Score (MCS): The average expression prevalence of a cluster's
  top 5 de novo marker genes (excluding mitochondrial genes), measuring internal
  consistency.

This version is modified to ensure perfect consistency between quantitative scores
and their corresponding visualizations, and to produce clear, diagonal dot plots.
It also includes an optional two-step HVG selection method and a robust fix for
bolding legend text in saved figures.

MODIFIED: The ratio-based annotation logic now auto-detects column names from the
marker database CSV to handle different file formats automatically.
MODIFIED: Now calculates and saves the "Marker Capture Score" for the auto-ratio
annotation to quantify its confidence.
"""

import os
import sys
import argparse
import random
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import anndata
import celltypist
from celltypist import annotate, models
from sklearn.metrics import silhouette_score
from collections import defaultdict

# ==============================================================================
# === HELPER FUNCTIONS (Unchanged) ===
# ==============================================================================

def _bold_right_margin_legend(fig_path):
    """
    Finds the legend in the current matplotlib figure and makes its text bold.
    Then saves the figure to the specified path.
    This is a workaround for backends where legend_fontweight is ignored.
    """
    fig = plt.gcf()
    # Iterate through all axes in the figure
    for ax in fig.axes:
        # Get the legend for the current axis
        leg = ax.get_legend()
        if leg is not None:
            # Set the font weight for each text object in the legend
            for txt in leg.get_texts():
                txt.set_fontweight('bold')
    
    # Save the modified figure with tight bounding box to prevent cutoff
    fig.savefig(fig_path, dpi=plt.rcParams['savefig.dpi'], bbox_inches='tight')


def reformat_dotplot_data(
    fraction_df: pd.DataFrame, top_genes_df: pd.DataFrame, output_dir: str,
    output_prefix: str, groupby_key: str
):
    """Reformats dot plot fraction data to a gene-centric sparse table."""
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

def extract_fraction_data_and_calculate_mcs(
    adata: anndata.AnnData, output_dir: str, output_prefix: str,
    groupby_key: str, top_genes_df: pd.DataFrame
):
    """Calculates and saves expression fractions and the MCS using a pre-filtered list of top marker genes."""
    print(f"[INFO] Calculating MCS and expression fractions for '{groupby_key}'...")
    if groupby_key not in adata.obs.columns:
        print(f"[ERROR] Grouping key '{groupby_key}' not found in adata.obs. Skipping MCS calculation.")
        return None

    unique_top_genes = top_genes_df['names'].unique().tolist()
    data_df = sc.get.obs_df(adata, keys=[groupby_key] + unique_top_genes, use_raw=(adata.raw is not None))
    fraction_df = data_df.groupby(groupby_key).apply(lambda x: (x[unique_top_genes] > 0).mean())

    mcs_scores = {}
    for cell_type in top_genes_df['group'].unique():
        markers_for_type = top_genes_df[top_genes_df['group'] == cell_type]['names']
        prevalence_values = fraction_df.loc[cell_type, markers_for_type]
        mcs_scores[cell_type] = prevalence_values.mean()

    mcs_df = pd.DataFrame.from_dict(mcs_scores, orient='index', columns=['MCS'])
    mcs_df.index.name = 'Cell_Type'
    mcs_csv_path = os.path.join(output_dir, f"{output_prefix}_marker_concordance_scores.csv")
    mcs_df.to_csv(mcs_csv_path)
    print(f"       -> Saved MCS scores to: {mcs_csv_path}")

    output_csv_path = os.path.join(output_dir, f"{output_prefix}_dotplot_fractions_{groupby_key}.csv")
    fraction_df.to_csv(output_csv_path)
    print(f"       -> Saved full fraction data to: {output_csv_path}")
    reformat_dotplot_data(fraction_df, top_genes_df, output_dir, output_prefix, groupby_key)

    return mcs_df

# ==============================================================================
# === MAIN ANALYSIS PIPELINE ===
# ==============================================================================

def main(args):
    """Main function to run the entire analysis pipeline."""
    print("--- Initializing CAS-MCS Scoring Pipeline ---")
    
    # --- Step 0: Setup and Reproducibility (Unchanged) ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    sc.settings.njobs = 1
    print(f"[INFO] Global random seed set to: {args.seed}")
    print("[INFO] Single-threaded execution enforced for reproducibility.")

    sc.settings.verbosity = 3
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False, dpi_save=args.fig_dpi)

    os.makedirs(args.output_dir, exist_ok=True)
    sc.settings.figdir = args.output_dir
    print(f"[INFO] Scanpy version: {sc.__version__}")
    print(f"[INFO] Output directory: {os.path.abspath(args.output_dir)}")

    # --- Step 1: Load Data (Unchanged) ---
    print("\n--- Step 1: Loading Data ---")
    adata = sc.read_10x_mtx(args.data_dir, var_names='gene_symbols', cache=True)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    print(f"       -> Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # --- Step 2: QC & Filtering (Unchanged) ---
    print("\n--- Step 2: Quality Control and Filtering ---")
    MITO_REGEX_PATTERN = r'^(MT|Mt|mt)[-._:]'
    adata.var['mt'] = adata.var_names.str.contains(MITO_REGEX_PATTERN, regex=True)
    print(f"       -> Identified {adata.var['mt'].sum()} mitochondrial genes using robust regex.")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    fig_qc_before, axs = plt.subplots(1, 4, figsize=(18, 4))
    sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, ax=axs[0], show=False)
    sc.pl.violin(adata, 'total_counts', jitter=0.4, ax=axs[1], show=False)
    sc.pl.violin(adata, 'pct_counts_mt', jitter=0.4, ax=axs[2], show=False)
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt', ax=axs[3], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.prefix}_qc_plots_before_filtering.png"))
    plt.close()

    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_cells(adata, max_genes=args.max_genes)
    adata = adata[adata.obs.pct_counts_mt < args.max_pct_mt, :]
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    print(f"       -> Filtered dims: {adata.n_obs} cells, {adata.n_vars} genes")

    # --- Step 3: Normalization, HVG, and Scaling (Unchanged) ---
    print("\n--- Step 3: Normalization, HVG, Scaling ---")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    if all(p is not None for p in [args.hvg_min_mean, args.hvg_max_mean, args.hvg_min_disp]):
        print("[INFO] Using two-step sequential HVG selection.")
        
        print(f"       Step 1: Filtering with min_mean={args.hvg_min_mean}, max_mean={args.hvg_max_mean}, min_disp={args.hvg_min_disp}")
        sc.pp.highly_variable_genes(adata, min_mean=args.hvg_min_mean, max_mean=args.hvg_max_mean, min_disp=args.hvg_min_disp)
        n_passed_threshold = adata.var.highly_variable.sum()
        print(f"       -> Found {n_passed_threshold} genes passing the thresholds.")
        
        print(f"       Step 2: Ranking candidates and selecting top {args.n_hvgs} genes.")
        hvg_df = adata.var[adata.var.highly_variable].sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:args.n_hvgs]
        
        adata.var['highly_variable'] = False
        adata.var.loc[top_genes, 'highly_variable'] = True
    else:
        print(f"[INFO] Using rank-based HVG selection with n_top_genes={args.n_hvgs}")
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs, flavor='seurat_v3')

    sc.pl.highly_variable_genes(adata, save=f"_{args.prefix}_hvg_plot.png", show=False)
    plt.close()
    
    adata = adata[:, adata.var.highly_variable]
    print(f"       -> Final selection: {adata.n_vars} highly variable genes for downstream analysis.")
    sc.pp.scale(adata, max_value=10)

    # --- Step 4: PCA, Clustering, UMAP & QC Score (Unchanged) ---
    print("\n--- Step 4: Dimensionality Reduction and Clustering ---")
    N_PCS_FOR_PCA = min(105, adata.n_vars - 1)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=N_PCS_FOR_PCA, random_state=args.seed)
    sc.pl.pca_variance_ratio(adata, log=True, n_pcs=N_PCS_FOR_PCA, save=f"_{args.prefix}_pca_variance.png", show=False)
    plt.close()

    n_pcs_to_use = min(args.n_pcs, adata.obsm['X_pca'].shape[1])
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=n_pcs_to_use, random_state=args.seed)
    sc.tl.leiden(adata, resolution=args.resolution, random_state=args.seed)
    sc.tl.umap(adata, random_state=args.seed)

    print("       -> Computing t-SNE embedding...")
    sc.tl.tsne(adata, n_pcs=n_pcs_to_use, random_state=args.seed)

    silhouette_avg = silhouette_score(adata.obsm['X_pca'][:, :n_pcs_to_use], adata.obs['leiden'])
    print(f"       -> Average Silhouette Score for Leiden clustering: {silhouette_avg:.3f}")
    
    n_labels_leiden = adata.obs['leiden'].nunique()
    sc.pl.umap(adata, color='leiden', legend_fontweight='bold', legend_loc='on data', title=f'UMAP of Leiden Clusters ({n_labels_leiden} clusters)\nSilhouette: {silhouette_avg:.3f}', palette=sc.pl.palettes.godsnot_102, save=f"_{args.prefix}_umap_leiden.png", show=False, size=10)
    plt.close()

    sc.pl.tsne(adata, color='leiden', legend_fontweight='bold', legend_loc='on data', title=f't-SNE of Leiden Clusters ({n_labels_leiden} clusters)\nSilhouette: {silhouette_avg:.3f}', palette=sc.pl.palettes.godsnot_102, save=f"_{args.prefix}_tsne_leiden.png", show=False, size=10)
    plt.close()

    # --- Step 5: CellTypist Annotation (CAS) (Unchanged) ---
    print("\n--- Step 5: CellTypist Annotation and CAS Calculation ---")
    model = models.Model.load(args.celltypist_model)
    
    print("[INFO] Annotating cells using the full log-normalized transcriptome (from adata.raw)...")
    predictions = celltypist.annotate(adata.raw.to_adata(), model=model, majority_voting=False)
    
    adata.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels'].astype('category')
    if 'conf_score' in predictions.predicted_labels.columns:
        adata.obs['ctpt_confidence'] = predictions.predicted_labels['conf_score']

    n_labels_individual = adata.obs['ctpt_individual_prediction'].nunique()
    sc.pl.umap(adata, color='ctpt_individual_prediction', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Per-Cell CellTypist Annotation ({n_labels_individual} types)', show=False, size=10)
    fig_path = os.path.join(args.output_dir, f"{args.prefix}_umap_per_cell_celltypist.png")
    _bold_right_margin_legend(fig_path)
    plt.close()

    cluster2label = adata.obs.groupby('leiden')['ctpt_individual_prediction'].agg(lambda x: x.value_counts().idxmax()).to_dict()
    adata.obs['ctpt_consensus_prediction'] = adata.obs['leiden'].map(cluster2label).astype('category')
    
    n_labels_consensus = adata.obs['ctpt_consensus_prediction'].nunique()
    sc.pl.umap(adata, color='ctpt_consensus_prediction', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Cluster-Consensus CellTypist Annotation ({n_labels_consensus} types)', show=False, size=10)
    fig_path = os.path.join(args.output_dir, f"{args.prefix}_cluster_celltypist_umap.png")
    _bold_right_margin_legend(fig_path)
    plt.close()

    purity_results = [{"Consensus_Cell_Type": name, "Total_Cells_in_Cluster": len(group), "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == name).sum(), "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == name).sum() / len(group) if len(group) > 0 else 0} for name, group in adata.obs.groupby('ctpt_consensus_prediction')]
    cas_df = pd.DataFrame(purity_results).sort_values(by="Total_Cells_in_Cluster", ascending=False)
    cas_output_path = os.path.join(args.output_dir, f"{args.prefix}_cluster_annotation_scores.csv")
    cas_df.to_csv(cas_output_path, index=False)
    print(f"       -> Saved CAS (Purity) scores to: {cas_output_path}")

    # --- Step 6: Marker Gene Analysis (MCS) & Plotting (Unchanged) ---
    print("\n--- Step 6: Marker Gene Analysis and MCS Calculation ---")
    marker_groupby_key = 'ctpt_consensus_prediction'
    top_genes_df, mcs_df = None, None
    
    label_counts = adata.obs[marker_groupby_key].value_counts()
    valid_labels = label_counts[label_counts > 1].index.tolist()
    
    if len(valid_labels) < 2:
        print(f"[WARNING] Skipping marker gene analysis: Fewer than 2 consensus groups with >1 cell.")
    else:
        marker_key = f"wilcoxon_{marker_groupby_key}"
        sc.tl.rank_genes_groups(adata, marker_groupby_key, groups=valid_labels, method='wilcoxon', use_raw=True, key_added=marker_key)
        
        marker_df = sc.get.rank_genes_groups_df(adata, key=marker_key, group=None)
        mito_prefixes_regex = r'^(mt|mt-|mt\.|mt_)'
        filtered_rows = []
        for grp, sub in marker_df.groupby('group', sort=False):
            non_mito_sub = sub[~sub['names'].str.lower().str.match(mito_prefixes_regex)]
            filtered_rows.append(non_mito_sub.head(5))
        top_genes_df = pd.concat(filtered_rows, ignore_index=True)

        with plt.rc_context({'font.size': 18, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
            genes_to_plot = top_genes_df.groupby('group')['names'].apply(list).to_dict()
            sc.pl.dotplot(adata, var_names=genes_to_plot, groupby=marker_groupby_key, categories_order=list(genes_to_plot.keys()), use_raw=True, save=f"_{args.prefix}_markers_celltypist_dotplot.png", show=False)
            plt.close()
        
        mcs_df = extract_fraction_data_and_calculate_mcs(adata, args.output_dir, args.prefix, marker_groupby_key, top_genes_df)

        if mcs_df is not None and top_genes_df is not None:
            top_genes_agg = top_genes_df.groupby('group')['names'].apply(', '.join).reset_index()
            top_genes_agg.rename(columns={'names': 'Top_5_Markers', 'group': 'Cell_Type'}, inplace=True)
            combined_df = pd.merge(mcs_df, top_genes_agg, on='Cell_Type')
            combined_df[['Cell_Type', 'MCS', 'Top_5_Markers']].to_csv(os.path.join(args.output_dir, f"{args.prefix}_mcs_and_top_markers.csv"), index=False)
            print(f"       -> Saved combined MCS and Top Markers.")

    # --- Step 7: Optional Annotations ---
    print("\n--- Step 7: Applying Other Annotation Strategies ---")
    if args.manual_map_csv:
        try:
            map_df = pd.read_csv(args.manual_map_csv)
            manual_map = dict(zip(map_df['leiden_cluster'].astype(str), map_df['cell_type']))
            adata.obs['cell_type_manual'] = adata.obs['leiden'].map(manual_map).astype('category')
            sc.pl.umap(adata, color='cell_type_manual', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Manual Cluster Annotation', show=False, size=10)
            fig_path = os.path.join(args.output_dir, f"{args.prefix}_umap_manual_annotation.png")
            _bold_right_margin_legend(fig_path)
            plt.close()
        except Exception as e:
            print(f"[WARNING] Could not apply manual annotation. Error: {e}")

    # <<< START OF MODIFIED SECTION: "SMART" RATIO-BASED ANNOTATION & MARKER CAPTURE SCORE >>>
    if args.marker_db_csv:
        try:
            print("[INFO] Performing smart ratio-based annotation...")
            sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', use_raw=True, key_added="wilcoxon_leiden")
            
            # --- Auto-detection logic ---
            type_col, gene_col = None, None
            header = pd.read_csv(args.marker_db_csv, nrows=0).columns.tolist()

            if 'cell_name' in header and 'Symbol' in header:
                type_col, gene_col = 'cell_name', 'Symbol'
            elif 'Cell Type' in header and 'Cell Marker' in header:
                type_col, gene_col = 'Cell Type', 'Cell Marker'
            
            if not type_col or not gene_col:
                raise ValueError("Could not auto-detect marker DB columns. Expected ('cell_name', 'Symbol') OR ('Cell Type', 'Cell Marker').")
            print(f"       -> Auto-detected format: TYPE='{type_col}', GENE='{gene_col}'")

            # --- Aggregation logic using the detected columns ---
            print("[INFO] Aggregating markers from marker database CSV...")
            db_df = pd.read_csv(args.marker_db_csv)
            db_markers_dict = defaultdict(set)
            
            for _, row in db_df.iterrows():
                if gene_col in row and type_col in row and isinstance(row[gene_col], str):
                    cell_type = row[type_col]
                    markers = {m.strip().upper() for m in str(row[gene_col]).split(',')}
                    db_markers_dict[cell_type].update(markers)
            
            print(f"       -> Correctly aggregated markers for {len(db_markers_dict)} unique cell types.")
            
            # --- Scoring and Annotation (using the aggregated dictionary) ---
            marker_names = adata.uns['wilcoxon_leiden']['names']
            auto_map = {}
            for cluster_id in adata.obs['leiden'].cat.categories:
                cluster_degs_for_ratio = {str(g).upper() for g in marker_names[cluster_id][:25]}
                
                best_score = -1
                best_cell_type = f"Unknown_{cluster_id}"
                
                for ct, ref_genes in db_markers_dict.items():
                    if not ref_genes: continue
                    intersection_size = len(cluster_degs_for_ratio.intersection(ref_genes))
                    score = intersection_size / len(ref_genes)
                    
                    if score > best_score:
                        best_score = score
                        best_cell_type = ct
                
                auto_map[cluster_id] = best_cell_type

            adata.obs['cell_type_auto_ratio'] = adata.obs['leiden'].map(auto_map).astype('category')
            sc.pl.umap(adata, color='cell_type_auto_ratio', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Ratio-Based Annotation', show=False, size=10)
            fig_path = os.path.join(args.output_dir, f"{args.prefix}_umap_ratio_based_annotation.png")
            _bold_right_margin_legend(fig_path)
            plt.close()

            # --- Marker Capture Score Calculation ---
            print("[INFO] Calculating Marker Capture Score for auto-ratio annotations...")
            score_results = []
            leiden_degs_structured = adata.uns['wilcoxon_leiden']['names']

            for cluster_id, assigned_label in auto_map.items():
                if pd.isna(assigned_label) or assigned_label.startswith("Unknown"):
                    continue
                
                reference_genes = db_markers_dict.get(assigned_label, set())
                total_reference_genes = len(reference_genes)

                if total_reference_genes == 0: continue

                cluster_degs_for_capture = {g.upper() for g in leiden_degs_structured[cluster_id][:args.n_degs_for_capture]}
                
                captured_genes = cluster_degs_for_capture.intersection(reference_genes)
                num_captured = len(captured_genes)
                
                capture_score = (num_captured / total_reference_genes) * 100 if total_reference_genes > 0 else 0
                
                score_results.append({
                    "Cluster_ID": cluster_id,
                    "Assigned_Cell_Type": assigned_label,
                    "Marker_Capture_Score (%)": capture_score,
                    "Captured_Genes_Count": num_captured,
                    "Total_Reference_Genes": total_reference_genes,
                    "Captured_Genes_List": ", ".join(sorted(list(captured_genes)))
                })

            if score_results:
                capture_score_df = pd.DataFrame(score_results).sort_values(by="Marker_Capture_Score (%)", ascending=False)
                score_output_path = os.path.join(args.output_dir, f"{args.prefix}_auto_ratio_marker_capture_scores.csv")
                capture_score_df.to_csv(score_output_path, index=False)
                print(f"       -> Saved Marker Capture Scores to: {score_output_path}")
            else:
                print("[INFO] No marker capture scores were calculated.")

        except Exception as e:
            print(f"[WARNING] Could not perform ratio-based annotation or score calculation. Error: {e}")
    # <<< END OF MODIFIED SECTION >>>

    # --- Step 8: Export Results (Unchanged) ---
    print("\n--- Step 8: Exporting All Results ---")
    cols_to_save = [col for col in ['leiden', 'ctpt_individual_prediction', 'ctpt_confidence', 'ctpt_consensus_prediction', 'cell_type_manual', 'cell_type_auto_ratio'] if col in adata.obs.columns]
    annotations_path = os.path.join(args.output_dir, f"{args.prefix}_all_annotations.csv")
    adata.obs[cols_to_save].to_csv(annotations_path)
    print(f"       -> All cell annotations saved to: {annotations_path}")
    final_adata_path = os.path.join(args.output_dir, f"{args.prefix}_final_processed.h5ad")
    adata.write(final_adata_path)
    print(f"       -> Final AnnData object saved to: {final_adata_path}")

    # --- Step 9: Final Metrics Verification (Unchanged) ---
    print("\n--- Step 9: Verifying Metrics Against Optimization Run ---")
    total_cells = len(adata.obs)
    total_matching = (adata.obs['ctpt_individual_prediction'].astype(str) == adata.obs['ctpt_consensus_prediction'].astype(str)).sum()
    weighted_cas = (total_matching / total_cells) * 100 if total_cells > 0 else 0.0
    cas_per_cluster = [(g['ctpt_individual_prediction'].astype(str) == g['ctpt_consensus_prediction'].astype(str).iloc[0]).mean() * 100 for _, g in adata.obs.groupby('leiden')]
    simple_cas = np.mean(cas_per_cluster) if cas_per_cluster else 0.0
    mean_mcs = mcs_df['MCS'].mean() * 100 if mcs_df is not None and not mcs_df.empty else 0.0
    n_individual_labels = adata.obs['ctpt_individual_prediction'].nunique()
    n_consensus_labels = adata.obs['ctpt_consensus_prediction'].nunique()
    
    target_map = {'simple_cas': "Simple Mean CAS", 'weighted_cas': "Weighted Mean CAS", 'mcs': "Mean MCS"}
    
    print("\n" + "="*50)
    print("--- Final Verification Summary ---")
    print(f"Optimization Target: {target_map.get(args.optimization_target, 'N/A')}")
    print(f"Random Seed Used: {args.seed}\n")
    print(f"Best n_hvg: {args.n_hvgs}")
    print(f"Best n_pcs: {args.n_pcs}")
    print(f"Best n_neighbors: {args.n_neighbors}")
    print(f"Best resolution: {args.resolution:.3f}\n")
    
    if args.optimization_target == 'simple_cas':
        print(f"Highest_simple_mean_cas_pct: {simple_cas:.2f}")
        print(f"Corresponding_weighted_mean_cas_pct: {weighted_cas:.2f}")
        print(f"Corresponding_mean_mcs_pct: {mean_mcs:.2f}\n")
    elif args.optimization_target == 'weighted_cas':
        print(f"Highest_weighted_mean_cas_pct: {weighted_cas:.2f}")
        print(f"Corresponding_simple_mean_cas_pct: {simple_cas:.2f}")
        print(f"Corresponding_mean_mcs_pct: {mean_mcs:.2f}\n")
    elif args.optimization_target == 'mcs':
        print(f"Highest_mean_mcs_pct: {mean_mcs:.2f}")
        print(f"Corresponding_weighted_mean_cas_pct: {weighted_cas:.2f}")
        print(f"Corresponding_simple_mean_cas_pct: {simple_cas:.2f}\n")
        
    print(f"Final_n_individual_labels: {n_individual_labels}")
    print(f"Final_n_consensus_labels: {n_consensus_labels}")
    print("="*50)

    print("\n--- CAS-MCS Scoring Pipeline Finished Successfully! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive command-line pipeline for single-cell analysis and quantitative annotation scoring (CAS & MCS).")
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--data_dir', type=str, required=True, help='Path to 10x Genomics data directory.')
    io_group.add_argument('--output_dir', type=str, required=True, help='Path to output directory.')
    io_group.add_argument('--celltypist_model', type=str, required=True, help='Path to CellTypist model file (.pkl).')
    io_group.add_argument('--prefix', type=str, default='sc_analysis_repro', help='Prefix for all output files.')
    io_group.add_argument('--marker_db_csv', type=str, default=None, help='(Optional) Path to a CSV with canonical markers for ratio-based annotation.')
    io_group.add_argument('--manual_map_csv', type=str, default=None, help='(Optional) Path to a CSV for direct manual cluster-to-name mapping.')
    
    qc_group = parser.add_argument_group('QC & Filtering')
    qc_group.add_argument('--min_genes', type=int, default=200, help='Min genes per cell.')
    qc_group.add_argument('--max_genes', type=int, default=7000, help='Max genes per cell.')
    qc_group.add_argument('--max_pct_mt', type=float, default=10.0, help='Max mitochondrial percentage.')
    qc_group.add_argument('--min_cells', type=int, default=3, help='Min cells per gene.')
    qc_group.add_argument('--mito_prefix', type=str, default='mt-', help='DEPRECATED, now uses robust regex. Kept for compatibility.')
    qc_group.add_argument('--hvg_min_mean', type=float, default=None, help='(Optional) Activates two-step HVG selection. Min mean for the initial filtering.')
    qc_group.add_argument('--hvg_max_mean', type=float, default=None, help='(Optional) Activates two-step HVG selection. Max mean for the initial filtering.')
    qc_group.add_argument('--hvg_min_disp', type=float, default=None, help='(Optional) Activates two-step HVG selection. Min dispersion for the initial filtering.')

    analysis_group = parser.add_argument_group('Analysis Parameters')
    analysis_group.add_argument('--n_hvgs', type=int, default=3000, help='Final number of top highly variable genes to select for downstream analysis.')
    analysis_group.add_argument('--n_pcs', type=int, default=80, help='Number of principal components.')
    analysis_group.add_argument('--n_neighbors', type=int, default=10, help='Number of neighbors.')
    analysis_group.add_argument('--resolution', type=float, default=2.0, help='Leiden clustering resolution.')
    # <<< START OF MODIFICATION: NEW ARGUMENT >>>
    analysis_group.add_argument('--n_degs_for_capture', type=int, default=50, help='Number of top DEGs for manual annotation Marker Capture Score.')
    # <<< END OF MODIFICATION >>>
    
    other_group = parser.add_argument_group('Other Settings')
    other_group.add_argument('--seed', default=42, type=int, help='Random seed.')
    other_group.add_argument('--fig_dpi', default=1000, type=int, help='Resolution (DPI) for saved figures.')
    other_group.add_argument('--optimization_target', type=str, default='simple_cas', choices=['simple_cas', 'weighted_cas', 'mcs'], help='Specifies optimization target for final report formatting.')

    args = parser.parse_args()
    main(args)