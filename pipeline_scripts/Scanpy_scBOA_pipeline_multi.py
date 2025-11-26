#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final, Reproducible Scanpy Analysis Script for Two-Sample Integration.

This script is designed to be run from the command line on a server. It takes
raw 10x data paths and a fixed set of optimal parameters (e.g., from a
Bayesian optimization run) to perform a complete, reproducible analysis. This
version has been unified with the single-sample verification script to ensure
perfect consistency in parameters, plotting aesthetics, and robustness.

Workflow:
1.  Load two 10x datasets (e.g., WT and Treated).
2.  Concatenate the datasets, adding 'sample' metadata.
3.  Perform standard quality control (QC) filtering with a robust mito regex.
4.  Normalize, log-transform, and identify highly variable genes (HVGs) per batch.
5.  Run PCA (with a dynamic component fix) and perform batch integration using Harmony.
6.  Build kNN graph, cluster with Leiden, and compute UMAP on integrated data.
7.  (Optional) Annotate cell types using a pre-trained CellTypist model.
    MODIFIED: Now outputs two UMAPs: one for per-cell predictions and one for cluster-consensus.
    MODIFIED: Now calculates and saves the Cluster Annotation Score (CAS) purity metrics.
8.  Find marker genes for raw Leiden clusters.
9.  (Optional) Perform manual annotation using a provided cell marker database.
    MODIFIED: This step now uses robust marker aggregation and auto-detects
    column names from the marker database CSV for improved flexibility.
    MODIFIED: Now calculates and saves the "Marker Capture Score" to quantify
    the confidence of the manual annotation.
10. Perform compositional analysis to see changes in cell type abundance.
11. Perform Differential Gene Expression (DGE) analysis between conditions.
12. Generate final summary plots (UMAPs, Heatmap, Dotplot) with consistent styling.
13. Save the final, fully processed AnnData object.

The use of a fixed random seed ensures that the output (clustering, UMAP, etc.)
is identical every time the script is run with the same inputs and parameters.

--- HOW TO USE EXAMPLE ---
$ python run_final_analysis_multi_sample_unified.py \
    --wt_path /path/to/wt/data/ \
    --treated_path /path/to/treated/data/ \
    --output_dir /path/to/your/output_folder/ \
    --output_prefix "MyExperiment_Final_Run" \
    --celltypist_model /path/to/Mouse_Whole_Brain.pkl \
    --cellmarker_db /path/to/marker_database.csv \
    --n_hvgs 3000 \
    --n_pcs 80 \
    --n_neighbors 10 \
    --resolution 2.0 \
    --seed 42 \
    --fig_dpi 1000 \
    --n_degs_for_capture 50
"""

# --- Step 0: Imports and Initial Setup ---
import os
import sys
import random
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server runs
import matplotlib.pyplot as plt
import anndata
# Added import for robust marker aggregation
from collections import defaultdict


# --- Robust imports for optional CellTypist and Harmony ---
try:
    from celltypist import annotate, models
    CELLTYPIST_INSTALLED = True
except ImportError:
    CELLTYPIST_INSTALLED = False

try:
    import harmonypy as hm
    HARMONY_INSTALLED = True
except ImportError:
    HARMONY_INSTALLED = False

# --- HELPER FUNCTIONS (UNIFIED WITH SCRIPT 1) ---

def _bold_right_margin_legend(fig_path):
    """
    Finds the legend in the current matplotlib figure and makes its text bold.
    Then saves the figure to the specified path. This is a workaround for
    backends where legend_fontweight is sometimes ignored for external legends.
    """
    fig = plt.gcf()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontweight('bold')
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

def extract_fraction_data_for_dotplot(
    adata: anndata.AnnData, output_dir: str, output_prefix: str,
    groupby_key: str, top_genes_df: pd.DataFrame
):
    """Calculates and saves expression fractions using a pre-filtered list of top marker genes."""
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


def main():
    """Main function to parse arguments and run the entire analysis pipeline."""
    # --- ARGUMENT PARSING (UNIFIED WITH VERIFIER SCRIPT) ---
    parser = argparse.ArgumentParser(
        description="Run a complete and reproducible scRNA-seq analysis pipeline for two conditions.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--wt_path', type=str, required=True, help="Path to the 10x directory for the control (WT) sample.")
    io_group.add_argument('--treated_path', type=str, required=True, help="Path to the 10x directory for the treated sample.")
    io_group.add_argument('--output_dir', type=str, required=True, help="Directory to save all output files.")
    io_group.add_argument('--output_prefix', type=str, default='sc_analysis_repro', help="Prefix for all output file names.")
    io_group.add_argument('--celltypist_model', type=str, default=None, help="Path to CellTypist model file (.pkl). If not provided, this step is skipped.")
    io_group.add_argument('--cellmarker_db', type=str, default=None, help="Path to a cell marker database (.csv) for optional manual annotation.")

    qc_group = parser.add_argument_group('QC & Filtering Parameters')
    qc_group.add_argument('--min_genes', type=int, default=200, help='Min genes per cell.')
    qc_group.add_argument('--max_genes', type=int, default=7000, help='Max genes per cell.')
    qc_group.add_argument('--max_pct_mt', type=float, default=10.0, help='Max mitochondrial percentage.')
    qc_group.add_argument('--min_cells', type=int, default=3, help='Min cells per gene.')
    qc_group.add_argument('--hvg_min_mean', type=float, default=None, help='(Optional) Activates two-step HVG. Min mean.')
    qc_group.add_argument('--hvg_max_mean', type=float, default=None, help='(Optional) Activates two-step HVG. Max mean.')
    qc_group.add_argument('--hvg_min_disp', type=float, default=None, help='(Optional) Activates two-step HVG. Min dispersion.')

    analysis_group = parser.add_argument_group('Core Analysis Parameters')
    analysis_group.add_argument('--n_hvgs', type=int, default=3000, help="Final number of top highly variable genes to select.")
    analysis_group.add_argument('--n_pcs_compute', type=int, default=105, help="Number of principal components to COMPUTE. Should be a large, fixed number.")
    analysis_group.add_argument('--n_pcs', type=int, default=80, help="Number of principal components to USE for neighborhood graph and UMAP.")
    analysis_group.add_argument('--n_neighbors', type=int, default=10, help="Number of neighbors (k) for the kNN graph.")
    analysis_group.add_argument('--resolution', type=float, default=2.0, help="Resolution for Leiden clustering.")
    analysis_group.add_argument('--n_top_genes', type=int, default=25, help="Number of top marker genes per cluster to use for annotation scoring.")
    # <<< START OF MODIFICATION: NEW ARGUMENT >>>
    analysis_group.add_argument('--n_degs_for_capture', type=int, default=50, help='Number of top DEGs to consider for the manual annotation Marker Capture Score.')
    # <<< END OF MODIFICATION >>>
    
    other_group = parser.add_argument_group('Other Settings')
    other_group.add_argument('--seed', default=42, type=int, help='Global random seed for reproducibility.')
    other_group.add_argument('--fig_dpi', default=1000, type=int, help='Resolution (DPI) for saved figures.')

    args = parser.parse_args()
    
    if args.n_pcs > args.n_pcs_compute:
        parser.error(f"--n_pcs ({args.n_pcs}) cannot be greater than --n_pcs_compute ({args.n_pcs_compute}).")

    # --- REPRODUCIBILITY AND SETUP ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    sc.settings.njobs = 1
    print(f"[INFO] Global random seed set to: {args.seed} for reproducibility.")
    print("[INFO] Single-threaded execution enforced.")

    SAMPLE_INFO = {'WT': {'path': args.wt_path}, 'Treated': {'path': args.treated_path}}
    OUTPUT_DIR, OUTPUT_PREFIX = args.output_dir, args.output_prefix
    FINAL_ANNOTATION_COLUMN = 'ctpt_consensus_prediction'
    CONDITION_OF_INTEREST, REFERENCE_CONDITION = 'Treated', 'WT'
    MITO_REGEX_PATTERN = r'^(MT|Mt|mt)[-._:]'

    # --- GLOBAL SCANPY SETTINGS ---
    sc.settings.verbosity = 3
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False, dpi_save=args.fig_dpi)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sc.settings.figdir = OUTPUT_DIR
    
    # --- Step 1 & 2: Loading & Concatenating ---
    print("\n--- Step 1 & 2: Loading and Concatenating Datasets ---")
    adatas = {sid: sc.read_10x_mtx(info['path'], var_names='gene_symbols', cache=True) for sid, info in SAMPLE_INFO.items()}
    for sid, adata_sample in adatas.items():
        adata_sample.var_names_make_unique(); adata_sample.obs['sample'] = sid
    adata = anndata.AnnData.concatenate(*adatas.values(), batch_key='sample', batch_categories=list(adatas.keys()))

    # --- Step 3: Quality Control ---
    print("\n--- Step 3: Quality Control ---")
    adata.var['mt'] = adata.var_names.str.contains(MITO_REGEX_PATTERN, regex=True)
    print(f"       -> Identified {adata.var['mt'].sum()} mitochondrial genes using robust regex.")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None, log1p=False)
    
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_cells(adata, max_genes=args.max_genes)
    adata = adata[adata.obs.pct_counts_mt < args.max_pct_mt, :]
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    print(f"       -> Filtered dims: {adata.n_obs} cells, {adata.n_vars} genes")

    # --- Step 4: Normalization, HVGs, Scaling ---
    print("\n--- Step 4: Normalization, HVG selection, Scaling ---")
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata); adata.raw = adata.copy()
    
    if all(p is not None for p in [args.hvg_min_mean, args.hvg_max_mean, args.hvg_min_disp]):
        print("[INFO] Using two-step sequential HVG selection.")
        
        print(f"       Step 1: Filtering with min_mean={args.hvg_min_mean}, max_mean={args.hvg_max_mean}, min_disp={args.hvg_min_disp}")
        sc.pp.highly_variable_genes(
            adata, min_mean=args.hvg_min_mean, max_mean=args.hvg_max_mean, 
            min_disp=args.hvg_min_disp, batch_key='sample'
        )
        n_passed_threshold = adata.var.highly_variable.sum()
        print(f"       -> Found {n_passed_threshold} genes passing the thresholds.")
        
        print(f"       Step 2: Ranking candidates and selecting top {args.n_hvgs} genes.")
        hvg_df = adata.var[adata.var.highly_variable].sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:args.n_hvgs]
        
        adata.var['highly_variable'] = False
        adata.var.loc[top_genes, 'highly_variable'] = True
    else:
        print(f"[INFO] Using rank-based HVG selection with n_top_genes={args.n_hvgs}")
        sc.pp.highly_variable_genes(
            adata, n_top_genes=args.n_hvgs, flavor='seurat_v3', batch_key='sample'
        )

    sc.pl.highly_variable_genes(adata, save=f"_{OUTPUT_PREFIX}_hvg_plot.png", show=False); plt.close()
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"       -> Final selection: {adata.n_vars} highly variable genes for downstream analysis.")
    sc.pp.scale(adata, max_value=10)

    # --- Step 5: PCA and Harmony Integration ---
    print("\n--- Step 5: PCA and Batch Correction with Harmony ---")
    n_pcs_to_compute = min(args.n_pcs_compute, adata.n_vars - 1)
    print(f"[INFO] Computing {n_pcs_to_compute} PCs from {adata.n_vars} HVGs (capped by --n_pcs_compute={args.n_pcs_compute}).")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs_to_compute, random_state=args.seed)
    
    pca_rep_key = 'X_pca'
    if HARMONY_INSTALLED:
        print("harmonypy is installed. Performing batch correction.")
        sc.external.pp.harmony_integrate(adata, key='sample', basis='X_pca', adjusted_basis='X_pca_harmony', random_state=args.seed)
        pca_rep_key = 'X_pca_harmony'
    else:
        print("[WARNING] harmonypy not found. Skipping Harmony integration.")

    # --- Step 6: Neighborhood Graph, Clustering, UMAP ---
    print("\n--- Step 6: Neighborhood, Clustering, and UMAP on Integrated Data ---")
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs, use_rep=pca_rep_key, random_state=args.seed)
    sc.tl.leiden(adata, resolution=args.resolution, random_state=args.seed)
    sc.tl.umap(adata, random_state=args.seed)
    sc.pl.umap(adata, color='sample', title='UMAP by Sample', save=f"_{OUTPUT_PREFIX}_umap_sample.png", show=False, size=10); plt.close()
    sc.pl.umap(adata, color='leiden', legend_loc='on data', legend_fontweight='bold', title=f'Leiden Clusters (res={args.resolution})', palette=sc.pl.palettes.godsnot_102, save=f"_{OUTPUT_PREFIX}_umap_leiden.png", show=False, size=10); plt.close()

    # --- Step 7: Annotate with CellTypist ---
    print("\n--- Step 7: Cell Type Annotation with CellTypist ---")
    if CELLTYPIST_INSTALLED and args.celltypist_model and os.path.exists(args.celltypist_model):
        model = models.Model.load(args.celltypist_model)
        print("[INFO] Annotating cells using the full log-normalized transcriptome (from adata.raw)...")
        predictions = annotate(adata.raw.to_adata(), model=model, majority_voting=False)
        adata.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels']
        
        print("[INFO] Plotting UMAP of individual per-cell CellTypist predictions...")
        sc.pl.umap(
            adata, color='ctpt_individual_prediction', title='Per-Cell CellTypist Annotation (Individual)',
            palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, size=10, show=False
        )
        per_cell_fig_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_per_cell_celltypist_umap.png")
        _bold_right_margin_legend(per_cell_fig_path); plt.close()
        print(f"       -> Saved per-cell annotation UMAP to: {per_cell_fig_path}")

        adata.obs[FINAL_ANNOTATION_COLUMN] = adata.obs.groupby('leiden')['ctpt_individual_prediction'].transform(lambda x: x.value_counts().idxmax()).astype('category')
        
        sc.pl.umap(adata, color=FINAL_ANNOTATION_COLUMN, title='Cluster-Consensus CellTypist Annotation', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, size=10, show=False)
        fig_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cluster_celltypist_umap.png"); _bold_right_margin_legend(fig_path); plt.close()
        
        # --- Calculate and Save Cluster Annotation Score (CAS) ---
        print("[INFO] Calculating and saving Cluster Annotation Scores (CAS)...")
        purity_results = [
            {
                "Consensus_Cell_Type": name,
                "Total_Cells_in_Cluster": len(group),
                "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == name).sum(),
                "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == name).sum() / len(group) if len(group) > 0 else 0
            }
            for name, group in adata.obs.groupby(FINAL_ANNOTATION_COLUMN)
        ]
        cas_df = pd.DataFrame(purity_results).sort_values(by="Total_Cells_in_Cluster", ascending=False)
        cas_output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cluster_annotation_scores.csv")
        cas_df.to_csv(cas_output_path, index=False)
        print(f"       -> Saved CAS (purity) scores to: {cas_output_path}")

        marker_key = f"wilcoxon_{FINAL_ANNOTATION_COLUMN}"
        sc.tl.rank_genes_groups(adata, FINAL_ANNOTATION_COLUMN, method='wilcoxon', use_raw=True, key_added=marker_key)
        
        marker_df = sc.get.rank_genes_groups_df(adata, key=marker_key, group=None)
        mito_prefixes_regex = r'^(mt|mt-|mt\.|mt_)' # A simple regex for filtering
        filtered_rows = [sub[~sub['names'].str.lower().str.match(mito_prefixes_regex)].head(5) for grp, sub in marker_df.groupby('group', sort=False)]
        top_genes_df = pd.concat(filtered_rows, ignore_index=True)
        
        with plt.rc_context({'font.size': 18, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
            genes_to_plot = top_genes_df.groupby('group')['names'].apply(list).to_dict()
            sc.pl.dotplot(adata, var_names=genes_to_plot, groupby=FINAL_ANNOTATION_COLUMN, categories_order=list(genes_to_plot.keys()), use_raw=True, save=f"_{OUTPUT_PREFIX}_markers_celltypist_dotplot.png", show=False)
            plt.close()
        
        extract_fraction_data_for_dotplot(adata, OUTPUT_DIR, OUTPUT_PREFIX, FINAL_ANNOTATION_COLUMN, top_genes_df)
    else:
        print("[INFO] CellTypist not run. Using Leiden clusters for downstream analysis.")
        adata.obs[FINAL_ANNOTATION_COLUMN] = adata.obs['leiden'].astype('category')

    # --- Step 8: Find Marker Genes for raw Leiden clusters ---
    print("\n--- Step 8: Calculating Marker Genes for Leiden Clusters ---")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', use_raw=True, key_added='wilcoxon_leiden')
    sc.pl.rank_genes_groups(adata, n_genes=20, key='wilcoxon_leiden', sharey=False, save=f"_{OUTPUT_PREFIX}_markers_leiden.png", show=False); plt.close()

    # --- Step 9: Manual Annotation with Marker DB (Optional) ---
    print("\n--- Step 9: Manual Annotation ---")
    if args.cellmarker_db and os.path.exists(args.cellmarker_db):
        try:
            print(f"       -> Annotating using marker DB: {args.cellmarker_db}")
            
            # --- Auto-detection of columns ---
            header = pd.read_csv(args.cellmarker_db, nrows=0).columns.tolist()
            type_col, gene_col = None, None
            if 'cell_name' in header and 'Symbol' in header:
                type_col, gene_col = 'cell_name', 'Symbol'
            elif 'Cell Type' in header and 'Cell Marker' in header:
                type_col, gene_col = 'Cell Type', 'Cell Marker'
            
            if not type_col:
                raise ValueError("Marker DB must contain either ('cell_name', 'Symbol') or ('Cell Type', 'Cell Marker') columns.")
            print(f"       -> Auto-detected format: TYPE='{type_col}', GENE='{gene_col}'")

            # --- Robust aggregation of markers ---
            db_df = pd.read_csv(args.cellmarker_db)
            db_markers_dict = defaultdict(set)
            for _, row in db_df.iterrows():
                if gene_col in row and type_col in row and isinstance(row[gene_col], str):
                    cell_type = row[type_col]
                    markers = {m.strip().upper() for m in str(row[gene_col]).split(',')}
                    db_markers_dict[cell_type].update(markers)
            print(f"       -> Aggregated markers for {len(db_markers_dict)} unique cell types.")

            # --- Score clusters against the aggregated marker DB to assign labels---
            leiden_markers_df = sc.get.rank_genes_groups_df(adata, key='wilcoxon_leiden', group=None)
            cluster_annotations = {}
            
            for cluster in adata.obs['leiden'].cat.categories:
                cluster_genes = set(
                    leiden_markers_df[leiden_markers_df['group'] == cluster]
                    .head(args.n_top_genes)['names']
                    .str.upper()
                )
                
                scores = {}
                for cell_type, db_genes in db_markers_dict.items():
                    intersection = len(cluster_genes.intersection(db_genes))
                    # Jaccard score for assignment
                    score = intersection / (len(cluster_genes) + len(db_genes) - intersection) if (len(cluster_genes) + len(db_genes) - intersection) > 0 else 0
                    scores[cell_type] = score

                if scores:
                    best_cell_type = max(scores, key=scores.get)
                    if scores[best_cell_type] > 0:
                        cluster_annotations[cluster] = best_cell_type
                    else:
                        cluster_annotations[cluster] = f"Unknown_{cluster}"
                else:
                    cluster_annotations[cluster] = f"Unknown_{cluster}"

            # --- Apply annotations and save results ---
            adata.obs['manual_annotation'] = adata.obs['leiden'].map(cluster_annotations).astype('category')
            pd.DataFrame.from_dict(cluster_annotations, orient='index', columns=['AssignedType']).to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_leiden_to_manual_annotation.csv"))
            
            sc.pl.umap(adata, color='manual_annotation', title='Manual Cluster Annotation', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, size=10, show=False)
            fig_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_umap_manual_annotation.png"); _bold_right_margin_legend(fig_path); plt.close()
            
            # <<< START OF MODIFICATION: MARKER CAPTURE SCORE CALCULATION >>>
            print("[INFO] Calculating Marker Capture Score for manual annotations...")
            score_results = []
            leiden_degs_structured = adata.uns['wilcoxon_leiden']['names']

            for cluster_id, assigned_label in cluster_annotations.items():
                if pd.isna(assigned_label) or assigned_label.startswith("Unknown"):
                    continue
                
                reference_genes = db_markers_dict.get(assigned_label, set())
                total_reference_genes = len(reference_genes)

                if total_reference_genes == 0:
                    continue

                # Get top N DEGs for the cluster, ensuring case-insensitivity
                cluster_degs_for_capture = {g.upper() for g in leiden_degs_structured[cluster_id][:args.n_degs_for_capture]}
                
                # Find captured genes
                captured_genes = cluster_degs_for_capture.intersection(reference_genes)
                num_captured = len(captured_genes)
                
                # Calculate the Marker Capture Score
                capture_score = (num_captured / total_reference_genes) * 100 if total_reference_genes > 0 else 0
                
                score_results.append({
                    "Cluster_ID": cluster_id,
                    "Manual_Cell_Type": assigned_label,
                    "Marker_Capture_Score (%)": capture_score,
                    "Captured_Genes_Count": num_captured,
                    "Total_Reference_Genes": total_reference_genes,
                    "Captured_Genes_List": ", ".join(sorted(list(captured_genes)))
                })

            if score_results:
                capture_score_df = pd.DataFrame(score_results).sort_values(by="Marker_Capture_Score (%)", ascending=False)
                score_output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_manual_annotation_marker_capture_scores.csv")
                capture_score_df.to_csv(score_output_path, index=False)
                print(f"       -> Saved Marker Capture Scores to: {score_output_path}")
            else:
                print("[INFO] No marker capture scores were calculated.")
            # <<< END OF MODIFICATION >>>

        except Exception as e:
            print(f"[ERROR] Manual annotation failed. Reason: {e}")

    else:
        print("[INFO] Cell marker DB not provided or not found. Skipping manual annotation.")
    
    # --- Step 10: Compositional Analysis ---
    print("\n--- Step 10: Compositional Analysis ---")
    
    # This DataFrame already contains the raw cell counts
    composition_counts = pd.crosstab(adata.obs[FINAL_ANNOTATION_COLUMN], adata.obs['sample'])
    
    # This DataFrame contains the percentages
    composition_perc = composition_counts.div(composition_counts.sum(axis=0), axis=1) * 100
    
    # <<< START OF MODIFICATION >>>
    # Create a new DataFrame for the counts with renamed columns
    # This adds a '_count' suffix to each column name (e.g., 'WT' -> 'WT_count')
    counts_with_suffix = composition_counts.add_suffix('_count')
    
    # Join the percentage DataFrame with the renamed counts DataFrame
    # They share the same index, so the join is straightforward
    combined_composition_df = composition_perc.join(counts_with_suffix)
    
    # Save the new combined DataFrame to a CSV file
    output_path_comp = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_composition_percentages_and_counts.csv")
    combined_composition_df.to_csv(output_path_comp)
    print(f"       -> Saved combined composition percentages and counts to: {output_path_comp}")
    # <<< END OF MODIFICATION >>>
    
    # The rest of the plotting code remains unchanged
    fig, ax = plt.subplots(figsize=(12, 8))
    composition_perc.T.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_ylabel('Percentage of Cells'); ax.set_xlabel('Sample'); ax.set_title('Cell Type Composition by Sample')
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_composition_barchart.png")); plt.close()

    # --- Step 11: Differential Gene Expression (DGE) Analysis ---
    print("\n--- Step 11: DGE Analysis (Treated vs. WT within each cell type) ---")
    dge_results = []
    for cell_type in adata.obs[FINAL_ANNOTATION_COLUMN].cat.categories:
        print(f"  -> Running DGE for: {cell_type}")
        sub_adata = adata[(adata.obs[FINAL_ANNOTATION_COLUMN] == cell_type)].copy()
        if len(sub_adata.obs['sample'].unique()) < 2:
            print(f"     [SKIP] Not enough conditions for DGE in '{cell_type}'.")
            continue
        try:
            sc.tl.rank_genes_groups(sub_adata, 'sample', groups=[CONDITION_OF_INTEREST], reference=REFERENCE_CONDITION, method='wilcoxon', use_raw=True, key_added='dge_result')
            dge_df = sc.get.rank_genes_groups_df(sub_adata, key='dge_result', group=CONDITION_OF_INTEREST)
            dge_df['cell_type'] = cell_type
            dge_results.append(dge_df)
        except Exception as e: print(f"     [ERROR] DGE failed for '{cell_type}'. Reason: {e}")
    if dge_results:
        full_dge_df = pd.concat(dge_results, ignore_index=True)
        full_dge_df.to_csv(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_DGE_Treated_vs_WT_by_celltype.csv"), index=False)
        print("DGE analysis complete. Full results saved.")
    else: print("No DGE results were generated.")
    
    # --- Step 12: Final Marker Heatmap ---
    print("\n--- Step 12: Generating Final Marker Heatmap ---")
    marker_key = f"wilcoxon_{FINAL_ANNOTATION_COLUMN}"
    if marker_key in adata.uns:
        genes_to_plot_list = []
        if 'top_genes_df' in locals() and not top_genes_df.empty:
            genes_to_plot_list = top_genes_df['names'].unique().tolist()
            print(f"       -> Generating heatmap with top non-mitochondrial marker genes per cell type.")
        else: 
            marker_genes_df = sc.get.rank_genes_groups_df(adata, key=marker_key, group=None)
            top_markers = marker_genes_df.groupby('group').head(5) # Fallback to 5 genes
            genes_to_plot_list = top_markers['names'].unique().tolist()
            print(f"       -> Generating heatmap with top 5 marker genes per cell type (fallback).")

        sc.pl.heatmap(adata, var_names=genes_to_plot_list, groupby=FINAL_ANNOTATION_COLUMN, show=False, dendrogram=True, save=f"_{OUTPUT_PREFIX}_top_markers_heatmap.png"); plt.close()
    else:
         print(f"[WARNING] Marker key '{marker_key}' not found. Cannot generate heatmap.")

    # --- Step 13: Saving Final AnnData Object ---
    print("\n--- Step 13: Saving Final AnnData Object ---")
    final_adata_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_final_processed.h5ad")
    try:
        adata.write(final_adata_path)
        print(f"       -> Final annotated AnnData object saved to: {final_adata_path}")
    except Exception as e: print(f"[ERROR] Could not save the final AnnData object: {e}")

    # --- FINAL VERIFICATION SUMMARY ---
    print("\n" + "="*50)
    print("--- Final Parameters Summary ---")
    print(f"Random Seed Used: {args.seed}\n")
    print(f"Final n_hvgs: {args.n_hvgs}")
    print(f"Final n_pcs (used): {args.n_pcs}")
    print(f"Final n_neighbors: {args.n_neighbors}")
    print(f"Final resolution: {args.resolution:.3f}\n")
    print(f"Final_n_leiden_clusters: {adata.obs['leiden'].nunique()}")
    if FINAL_ANNOTATION_COLUMN in adata.obs.columns:
        print(f"Final_n_consensus_labels: {adata.obs[FINAL_ANNOTATION_COLUMN].nunique()}")
    print("="*50)

    print("\n--- MULTI-SAMPLE ANALYSIS PIPELINE COMPLETE ---")

if __name__ == '__main__':
    main()