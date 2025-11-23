#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-Line Interface for the scBOA Pipeline.

This script handles argument parsing and serves as the entry point for the
scBOA pipeline when it's installed as a package. It imports and calls the 
main orchestrator function from the `pipeline` module.
"""

import argparse

# This is the crucial relative import. It looks for a file named 'pipeline.py'
# within the same package directory ('src/scboa/') and imports the 'main' function.
from .pipeline import main

def run():
    """
    Parses all command-line arguments and executes the main scBOA pipeline.
    
    This function is the target for the [project.scripts] entry point specified
    in the pyproject.toml file.
    """
    parser = argparse.ArgumentParser(
        description="Integrated Two-Stage Bayesian Optimization and Final Analysis Pipeline for scRNA-seq.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    stage1_group = parser.add_argument_group('Stage 1 & 2: Main I/O and Mode')
    mode_group = stage1_group.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--data_dir', type=str, help='Path to 10x Genomics data for single-sample analysis.')
    mode_group.add_argument('--multi_sample', nargs=2, metavar=('WT_DIR', 'TREATED_DIR'), help='Two paths for WT/Control and Treated/Perturbed 10x data for multi-sample integration.')
    stage1_group.add_argument('--output_dir', type=str, required=True, help='Path for all output files.')
    stage1_group.add_argument('--model_path', type=str, required=True, help='Path to CellTypist model (.pkl).')
    stage1_group.add_argument('--output_prefix', type=str, default='bayesian_opt', help='Base prefix for Stage 1 output files.')

    opt_group = parser.add_argument_group('Stage 1: Optimization Parameters')
    opt_group.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility.')
    opt_group.add_argument('--n_calls', type=int, default=50, help='Number of trials for EACH of the three optimization strategies.')
    opt_group.add_argument(
        '--model_type',
        type=str,
        default='structural',
        choices=['biological', 'structural', 'silhouette'],
        help= ("'biological': balances CAS & MCS.\n"
               "'structural' (default): adds silhouette score to balance biological concordance with cluster quality.\n"
               "'silhouette': optimizes solely to maximize the silhouette score.")
    )
    opt_group.add_argument('--marker_gene_model', type=str, default='non-mitochondrial', choices=['all', 'non-mitochondrial'], help="'all': use all genes. 'non-mitochondrial' (default): exclude mitochondrial genes from MCS markers.")
    opt_group.add_argument('--target', type=str, default='all', choices=['all', 'weighted_cas', 'simple_cas', 'mcs'], help="'all' (default): runs a single, balanced optimization. Other options optimize for that specific metric.")
    
    opt_group.add_argument(
        '--cas_aggregation_method',
        type=str,
        default='leiden',
        choices=['leiden', 'consensus'],
        help=("Method for calculating Simple Mean CAS and for determining refinement candidates.\n"
              "'leiden' (default): Averages the purity of each individual Leiden cluster.\n"
              "'consensus': Merges Leiden clusters with the same consensus label, then averages their purity.")
    )

    hvg_group = parser.add_argument_group('Stage 1 & 2: HVG Selection Method')
    hvg_group.add_argument('--hvg_min_mean', type=float, default=None, help='(Optional) Activates two-step HVG selection. Min mean for initial filtering.')
    hvg_group.add_argument('--hvg_max_mean', type=float, default=None, help='(Optional) Activates two-step HVG selection. Max mean for initial filtering.')
    hvg_group.add_argument('--hvg_min_disp', type=float, default=None, help='(Optional) Activates two-step HVG selection. Min dispersion for initial filtering.')

    qc_group = parser.add_argument_group('Stage 1 & 2: QC & Filtering Parameters')
    qc_group.add_argument('--min_genes', type=int, default=200, help='Min genes per cell.')
    qc_group.add_argument('--max_genes', type=int, default=7000, help='Max genes per cell.')
    qc_group.add_argument('--max_pct_mt', type=float, default=10.0, help='Max mitochondrial percentage.')
    qc_group.add_argument('--min_cells', type=int, default=3, help='Min cells per gene.')

    stage2_group = parser.add_argument_group('Stage 2 & Optional Refinement: Final Run Parameters')
    stage2_group.add_argument('--final_run_prefix', type=str, default='sc_analysis_repro', help='Prefix for all output files in the Stage 2 subdirectory.')
    stage2_group.add_argument('--fig_dpi', default=500, type=int, help='Resolution (DPI) for saved figures in Stage 2.')
    stage2_group.add_argument('--n_pcs_compute', type=int, default=105, help="Number of principal components to COMPUTE in Stage 1 and 2.")
    stage2_group.add_argument('--n_top_genes', type=int, default=5, help="Number of top marker genes to show in plots/tables in Stage 1 and 2.")
    stage2_group.add_argument('--cellmarker_db', type=str, default=None, help="(Optional) Path to a cell marker database (.csv) for manual annotation in Stage 2.")
    stage2_group.add_argument('--n_degs_for_capture', type=int, default=50, help="Number of top DEGs per cluster to use for the Marker Capture Score calculation in Stage 2.")
    stage2_group.add_argument('--cas_refine_threshold', type=float, default=None, help="(Optional) CAS percentage threshold (0-100). If a cluster's CAS is below this, its cells are pooled for a second, refined optimization run.")
    stage2_group.add_argument('--refinement_depth', type=int, default=1, help="(Optional) Maximum number of times to repeat the refinement process on failing cells. Default is 1.")
    stage2_group.add_argument('--min_cells_refinement', type=int, default=100, help="(Optional) Minimum number of failing cells required to trigger a refinement loop. Default is 100.")

    # Parse the arguments provided by the user from the command line
    parsed_args = parser.parse_args()

    # Apply any pre-processing logic to the arguments before calling the main function
    if parsed_args.multi_sample and "harmony" not in parsed_args.output_prefix:
        parsed_args.output_prefix += "_harmony"
    
    # Call the main orchestrator function from pipeline.py and pass the parsed arguments
    main(parsed_args)
