#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-Line Interface for the scBOA Pipeline.

This module handles argument parsing and serves as the package entry point for
the scBOA pipeline when installed via pyproject.toml:

    scboa = "scboa.cli:run"
"""

import argparse
from .pipeline import main

def run():
    """
    Parse command-line arguments and execute the main scBOA pipeline.
    """
    parser = argparse.ArgumentParser(
        description="scBOA: Integrated Two-Stage Bayesian Optimization and Final Analysis Pipeline for scRNA-seq.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ------------------------------------------------------------------
    # Main I/O and analysis mode
    # ------------------------------------------------------------------
    stage1_group = parser.add_argument_group('Stage 1 & 2: Main I/O and Mode')
    
    mode_group = stage1_group.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--data_dir', type=str, help='Path to 10x Genomics data for single-sample analysis.')
    mode_group.add_argument('--multi_sample', nargs=2, metavar=('WT_DIR', 'TREATED_DIR'), 
                            help='Two paths for WT/Control and Treated/Perturbed 10x data for multi-sample integration.')
    
    stage1_group.add_argument('--integration_method', type=str, default='harmony', choices=['harmony', 'scanorama', 'bbknn'],
                              help="Batch correction method to use when running in --multi_sample mode. (Default: harmony)")
    stage1_group.add_argument('--output_dir', type=str, required=True, help='Path for all output files.')
    stage1_group.add_argument('--model_path', type=str, required=True, help='Path to CellTypist model (.pkl).')
    stage1_group.add_argument('--output_prefix', type=str, default='bayesian_opt', help='Base prefix for Stage 1 output files.')
    stage1_group.add_argument('--threads', type=int, default=16, help='Number of threads/CPUs to use for parallel processing.')

    # ------------------------------------------------------------------
    # Optimization parameters
    # ------------------------------------------------------------------
    opt_group = parser.add_argument_group('Stage 1: Optimization Parameters')
    opt_group.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility.')
    opt_group.add_argument('--n_calls', type=int, default=50, help='Number of trials for EACH of the three optimization strategies.')
    opt_group.add_argument('--subsample_size', type=int, default=None, 
                           help="(Optional) Randomly subsample the dataset to this number of cells for Stage 1 optimization to drastically reduce runtime on large atlases.")
    
    opt_group.add_argument(
        '--model_type', type=str, default='structural', choices=['biological', 'structural', 'silhouette'],
        help="'biological': balances CAS & MCS.\n'structural' (default): adds silhouette score.\n'silhouette': optimizes solely to maximize the silhouette score."
    )
    opt_group.add_argument('--marker_gene_model', type=str, default='non-mitochondrial', choices=['all', 'non-mitochondrial'], 
                           help="'all': use all genes. 'non-mitochondrial' (default): exclude mitochondrial genes from MCS markers.")
    opt_group.add_argument(
        '--target', type=str, default='all', choices=['all', 'balanced', 'weighted_cas', 'simple_cas', 'mcs'],
        help="'all' (default): runs the balanced optimization. 'balanced': explicitly optimize the balanced objective. Other options optimize for that specific metric only."
    )
    opt_group.add_argument(
        '--benchmark_optimizer', type=str, default=None, choices=['gp', 'optuna', 'random'],
        help="(Optional) Optimizer backend for benchmarking.\n'gp' or None: default scBOA GP-BO backend.\n'optuna': Optuna TPE backend.\n'random': equal-budget random-search baseline."
    )
    opt_group.add_argument(
        '--cas_aggregation_method', type=str, default='leiden', choices=['leiden', 'consensus'],
        help="Method for calculating Simple Mean CAS and for determining refinement candidates.\n'leiden' (default) or 'consensus'."
    )

    # ------------------------------------------------------------------
    # HVG selection
    # ------------------------------------------------------------------
    hvg_group = parser.add_argument_group('Stage 1 & 2: HVG Selection Method')
    hvg_group.add_argument('--hvg_min_mean', type=float, default=None, help='(Optional) Activates two-step HVG selection. Min mean for initial filtering.')
    hvg_group.add_argument('--hvg_max_mean', type=float, default=None, help='(Optional) Activates two-step HVG selection. Max mean for initial filtering.')
    hvg_group.add_argument('--hvg_min_disp', type=float, default=None, help='(Optional) Activates two-step HVG selection. Min dispersion for initial filtering.')

    # ------------------------------------------------------------------
    # QC and filtering
    # ------------------------------------------------------------------
    qc_group = parser.add_argument_group('Stage 1 & 2: QC & Filtering Parameters')
    qc_group.add_argument('--min_genes', type=int, default=200, help='Min genes per cell.')
    qc_group.add_argument('--max_genes', type=int, default=7000, help='Max genes per cell.')
    qc_group.add_argument('--max_pct_mt', type=float, default=10.0, help='Max mitochondrial percentage.')
    qc_group.add_argument('--min_cells', type=int, default=3, help='Min cells per gene.')

    # ------------------------------------------------------------------
    # Stage 2 & Refinement
    # ------------------------------------------------------------------
    stage2_group = parser.add_argument_group('Stage 2 & Optional Refinement: Final Run Parameters')
    stage2_group.add_argument('--final_run_prefix', type=str, default='sc_analysis_repro', help='Prefix for all output files in the Stage 2 subdirectory.')
    stage2_group.add_argument('--fig_dpi', default=500, type=int, help='Resolution (DPI) for saved figures in Stage 2.')
    stage2_group.add_argument('--n_pcs_compute', type=int, default=105, help="Number of principal components to COMPUTE in Stage 1 and 2.")
    stage2_group.add_argument('--n_top_genes', type=int, default=5, help="Number of top marker genes to show in plots/tables in Stage 1 and 2.")
    
    stage2_group.add_argument('--reference_marker_db', type=str, default=None, help="(Optional) Path to a combined reference marker database (.csv) for manual annotation and F1 scoring.")
    stage2_group.add_argument('--marker_prior_species', type=str, default='Human', help="Species filter for marker DB (e.g., 'Human' or 'Mouse').")
    stage2_group.add_argument('--marker_prior_organ', type=str, default='Blood', help="Organ/tissue filter for marker DB (e.g., 'Blood', 'Peripheral Blood').")
    stage2_group.add_argument('--n_degs_for_capture', type=int, default=5, help="Number of top DEGs per cluster to use for the Marker Capture Score calculation in Stage 2.")
    stage2_group.add_argument('--cas_refine_threshold', type=float, default=None, help="(Optional) CAS percentage threshold (0-100). If a cluster's CAS is below this, its cells are pooled for a second, refined optimization run.")
    
    stage2_group.add_argument('--f1_db_celltype_col', type=str, default=None, help="(Optional) Column name in the marker DB CSV containing cell type names for F1 scoring. Auto-detected if not provided.")
    stage2_group.add_argument('--f1_db_gene_col', type=str, default=None, help="(Optional) Column name in the marker DB CSV containing marker genes for F1 scoring. Auto-detected if not provided.")
    stage2_group.add_argument('--f1_groupby_key', type=str, default='ctpt_consensus_prediction', choices=['ctpt_consensus_prediction', 'manual_annotation', 'leiden'], help="Grouping key used to compute F1. Default uses cell-annotated clusters (ctpt_consensus_prediction).")
    stage2_group.add_argument('--marker_score_metric', type=str, default='f1', choices=['f1', 'jaccard', 'capture'], help="Marker scoring metric for cell-type annotation. Default is F1.")
    
    stage2_group.add_argument('--refinement_depth', type=int, default=1, help="(Optional) Maximum number of times to repeat the refinement process on failing cells. Default is 1.")
    stage2_group.add_argument('--min_cells_refinement', type=int, default=100, help="(Optional) Minimum number of failing cells required to trigger a refinement loop. Default is 100.")
    
    stage2_group.add_argument('--use_f1', action='store_true', help="Include marker-based F1 in Stage 1 optimization score.")
    stage2_group.add_argument('--mps_bonus_weight', type=float, default=0.2, help="Additive bonus weight for the Marker Prior Score (F1). Final Score = Base Score + (mps_bonus_weight * F1). Default 0.2 = max 20% bonus.")
    stage2_group.add_argument('--use_confidence', action='store_true', help="Include the mean CellTypist annotation confidence score in the geometric mean calculation for the optimization target.")
    stage2_group.add_argument('--min_confidence', type=float, default=None, help="(Optional) Per-cell CellTypist confidence threshold in [0,1]. Cells with ctpt_confidence < this value are dropped from all downstream analysis (per sample).")
    
    stage2_group.add_argument('--compute_soft_cas', action='store_true', default=False, help="Compute optional soft-CAS metrics using the full CellTypist probability matrix. This provides a sensitivity analysis for probabilistic annotation concordance.")
    stage2_group.add_argument('--use_soft_cas', action='store_true', default=False, help="Include the soft-CAS dot-product score in the balanced optimization objective.")

    # ------------------------------------------------------------------
    # Stage 3: Multi-Sample Integration
    # ------------------------------------------------------------------
    stage3_group = parser.add_argument_group('Stage 3: Multi-Sample Integration')
    stage3_group.add_argument('--samples', nargs='+', default=None, metavar='NAME=PATH', help="(Stage 3) Two or more samples as NAME=PATH pairs, e.g. --samples WT=/path/wt Treated=/path/treated Drug=/path/drug.")
    stage3_group.add_argument('--sample_names', nargs='+', default=None, help="Space-separated sample names. Must be paired 1:1 with --sample_paths.")
    stage3_group.add_argument('--sample_paths', nargs='+', default=None, help="Space-separated sample data directories. Must be paired 1:1 with --sample_names.")
    stage3_group.add_argument('--enable_stage3_integration', action='store_true', help="Run the Stage 3 merge + Harmony integration + marker re-annotation pipeline.")
    stage3_group.add_argument('--integration_hvg_strategy', type=str, default='sample_specific_union', choices=['fixed_global', 'batch_consensus', 'sample_specific_union', 'sample_specific_weighted'], help="HVG selection strategy on the merged object for Stage 3.")
    stage3_group.add_argument('--integration_fixed_n_hvg', type=int, default=3000)
    stage3_group.add_argument('--integration_n_pcs', type=int, default=30)
    stage3_group.add_argument('--integration_n_neighbors', type=int, default=15)
    stage3_group.add_argument('--min_hvg_sample_recurrence', type=int, default=1, help="Min #samples a gene must be HVG in to enter the union (sample_specific_*).")
    stage3_group.add_argument('--integration_max_union_genes', type=int, default=5000)
    stage3_group.add_argument('--stage3_marker_topN', type=int, default=50, help="Top-N DE genes per integrated cell type used in marker DB F1 match.")

    # ------------------------------------------------------------------
    # Parse and validate logic
    # ------------------------------------------------------------------
    parsed_args = parser.parse_args()

    # Soft-CAS consistency logic
    if parsed_args.use_soft_cas:
        parsed_args.compute_soft_cas = True

    if parsed_args.use_soft_cas and parsed_args.target not in ['all', 'balanced']:
        print("[WARNING] --use_soft_cas was provided, but --target is not 'all' or 'balanced'. Soft-CAS will be computed and reported, but it will not affect the optimization objective.")

    # Merge --sample_names / --sample_paths into --samples NAME=PATH form
    if parsed_args.sample_names or parsed_args.sample_paths:
        if not (parsed_args.sample_names and parsed_args.sample_paths):
            parser.error("--sample_names and --sample_paths must be given together.")
        if len(parsed_args.sample_names) != len(parsed_args.sample_paths):
            parser.error(f"--sample_names ({len(parsed_args.sample_names)}) and --sample_paths ({len(parsed_args.sample_paths)}) must have equal length.")
        
        paired = [f"{n}={p}" for n, p in zip(parsed_args.sample_names, parsed_args.sample_paths)]
        parsed_args.samples = (parsed_args.samples or []) + paired

    # Validate that at least one main input mode is selected
    if not (parsed_args.data_dir or parsed_args.multi_sample or (parsed_args.samples and parsed_args.enable_stage3_integration)):
        parser.error("Must specify --data_dir, --multi_sample, or --samples NAME=PATH ... --enable_stage3_integration")

    # Basic validation of NAME=PATH sample syntax
    if parsed_args.samples:
        malformed = [s for s in parsed_args.samples if "=" not in s]
        if malformed:
            parser.error(f"Each --samples entry must use NAME=PATH format. Malformed entries: {malformed}")

    # Launch pipeline
    main(parsed_args)

if __name__ == "__main__":
    run()