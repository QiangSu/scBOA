#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-Line Interface for the scBOA Pipeline.

This module handles argument parsing and serves as the package entry point for
the scBOA pipeline when installed via pyproject.toml:

    scboa = "scboa.cli:run"

The CLI supports:
1. Single-sample analysis via --data_dir
2. Two-sample early integration via --multi_sample WT_DIR TREATED_DIR
3. Atlas-scale per-sample hierarchical integration via
   --samples NAME=PATH ... --enable_stage3_integration
4. Equal-budget optimizer benchmarking via --benchmark_optimizer
"""

import argparse

from .pipeline import main


def run():
    """
    Parse command-line arguments and execute the main scBOA pipeline.
    """

    parser = argparse.ArgumentParser(
        description=(
            "scBOA: Integrated Bayesian Optimization and Final Analysis "
            "Pipeline for single-cell RNA-seq."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Main I/O and analysis mode
    # ------------------------------------------------------------------
    io_group = parser.add_argument_group("Main I/O and analysis mode")

    mode_group = io_group.add_mutually_exclusive_group(required=False)

    mode_group.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to 10x Genomics data for single-sample analysis.",
    )

    mode_group.add_argument(
        "--multi_sample",
        nargs=2,
        metavar=("WT_DIR", "TREATED_DIR"),
        default=None,
        help=(
            "Two paths for early-integrated two-sample analysis, typically "
            "WT/control and treated/perturbed 10x data directories."
        ),
    )

    mode_group.add_argument(
        "--samples",
        nargs="+",
        metavar="NAME=PATH",
        default=None,
        help=(
            "Atlas-scale multi-sample input for hierarchical integration. "
            "Provide one or more NAME=PATH pairs, for example:\n"
            "  --samples WT_rep1=/data/wt1 WT_rep2=/data/wt2 HD_rep1=/data/hd1\n"
            "This mode requires --enable_stage3_integration."
        ),
    )

    io_group.add_argument(
        "--enable_stage3_integration",
        action="store_true",
        help=(
            "Enable per-sample hierarchical Stage 3 integration. "
            "Required when using --samples NAME=PATH inputs."
        ),
    )

    io_group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path for all output files.",
    )

    io_group.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to CellTypist model file (.pkl).",
    )

    io_group.add_argument(
        "--output_prefix",
        type=str,
        default="bayesian_opt",
        help="Base prefix for Stage 1 output files.",
    )

    # ------------------------------------------------------------------
    # Optimization parameters
    # ------------------------------------------------------------------
    opt_group = parser.add_argument_group("Stage 1: optimization parameters")

    opt_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )

    opt_group.add_argument(
        "--n_calls",
        type=int,
        default=50,
        help="Number of optimization trials.",
    )

    opt_group.add_argument(
        "--benchmark_optimizer",
        type=str,
        default="optuna",
        choices=["optuna", "random"],
        help=(
            "Optimizer used for equal-budget benchmarking.\n"
            "'optuna': Optuna/TPE Bayesian optimization.\n"
            "'random': equal-budget random search."
        ),
    )

    opt_group.add_argument(
        "--model_type",
        type=str,
        default="structural",
        choices=["biological", "structural", "silhouette"],
        help=(
            "'biological': balances CAS and MCS.\n"
            "'structural' default: balances biological concordance with "
            "cluster structure using silhouette score.\n"
            "'silhouette': optimizes only silhouette score."
        ),
    )

    opt_group.add_argument(
        "--marker_gene_model",
        type=str,
        default="non-mitochondrial",
        choices=["all", "non-mitochondrial"],
        help=(
            "'all': use all genes for marker capture scoring.\n"
            "'non-mitochondrial': exclude mitochondrial genes from MCS markers."
        ),
    )

    opt_group.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "weighted_cas", "simple_cas", "mcs"],
        help=(
            "'all': run balanced optimization.\n"
            "Other options optimize the selected metric."
        ),
    )

    opt_group.add_argument(
        "--cas_aggregation_method",
        type=str,
        default="leiden",
        choices=["leiden", "consensus"],
        help=(
            "Method for calculating Simple Mean CAS and identifying "
            "refinement candidates.\n"
            "'leiden': average purity across individual Leiden clusters.\n"
            "'consensus': merge Leiden clusters with the same consensus label "
            "before averaging purity."
        ),
    )

    opt_group.add_argument(
        "--soft_cas",
        action="store_true",
        help=(
            "Enable soft-CAS scoring using probabilistic CellTypist evidence "
            "when available."
        ),
    )

    opt_group.add_argument(
        "--min_confidence",
        type=float,
        default=None,
        help=(
            "Optional minimum CellTypist confidence threshold for label usage "
            "or confidence-aware scoring. If not provided, all labels are used."
        ),
    )

    # ------------------------------------------------------------------
    # HVG selection
    # ------------------------------------------------------------------
    hvg_group = parser.add_argument_group("HVG selection")

    hvg_group.add_argument(
        "--hvg_min_mean",
        type=float,
        default=None,
        help="Optional min mean for two-step HVG filtering.",
    )

    hvg_group.add_argument(
        "--hvg_max_mean",
        type=float,
        default=None,
        help="Optional max mean for two-step HVG filtering.",
    )

    hvg_group.add_argument(
        "--hvg_min_disp",
        type=float,
        default=None,
        help="Optional min dispersion for two-step HVG filtering.",
    )

    # ------------------------------------------------------------------
    # QC and filtering
    # ------------------------------------------------------------------
    qc_group = parser.add_argument_group("QC and filtering parameters")

    qc_group.add_argument(
        "--min_genes",
        type=int,
        default=200,
        help="Minimum number of genes per cell/nucleus.",
    )

    qc_group.add_argument(
        "--max_genes",
        type=int,
        default=7000,
        help="Maximum number of genes per cell/nucleus.",
    )

    qc_group.add_argument(
        "--max_pct_mt",
        type=float,
        default=10.0,
        help="Maximum mitochondrial percentage.",
    )

    qc_group.add_argument(
        "--min_cells",
        type=int,
        default=3,
        help="Minimum number of cells expressing a gene.",
    )

    qc_group.add_argument(
        "--enable_scrublet",
        action="store_true",
        help="Enable Scrublet-based doublet scoring/filtering if implemented.",
    )

    qc_group.add_argument(
        "--scrublet_threshold",
        type=float,
        default=None,
        help=(
            "Optional Scrublet doublet-score threshold. If not provided, "
            "Scrublet's automatic threshold is used when available."
        ),
    )

    # ------------------------------------------------------------------
    # Stage 2 final run and refinement
    # ------------------------------------------------------------------
    stage2_group = parser.add_argument_group(
        "Stage 2 and optional low-confidence refinement"
    )

    stage2_group.add_argument(
        "--final_run_prefix",
        type=str,
        default="sc_analysis_repro",
        help="Prefix for all Stage 2 output files.",
    )

    stage2_group.add_argument(
        "--fig_dpi",
        type=int,
        default=500,
        help="Resolution DPI for saved figures.",
    )

    stage2_group.add_argument(
        "--n_pcs_compute",
        type=int,
        default=105,
        help="Number of principal components to compute.",
    )

    stage2_group.add_argument(
        "--n_top_genes",
        type=int,
        default=5,
        help="Number of top marker genes to show in plots/tables.",
    )

    stage2_group.add_argument(
        "--cellmarker_db",
        type=str,
        default=None,
        help="Optional path to a cell marker database CSV for manual annotation.",
    )

    stage2_group.add_argument(
        "--n_degs_for_capture",
        type=int,
        default=50,
        help=(
            "Number of top DEGs per cluster used for Marker Capture Score "
            "calculation."
        ),
    )

    stage2_group.add_argument(
        "--cas_refine_threshold",
        type=float,
        default=None,
        help=(
            "Optional CAS percentage threshold, 0-100. Clusters below this "
            "threshold are pooled for refinement."
        ),
    )

    stage2_group.add_argument(
        "--refinement_depth",
        type=int,
        default=1,
        help="Maximum number of low-confidence refinement iterations.",
    )

    stage2_group.add_argument(
        "--min_cells_refinement",
        type=int,
        default=100,
        help="Minimum number of low-confidence cells required to trigger refinement.",
    )

    # ------------------------------------------------------------------
    # Stage 3 hierarchical integration
    # ------------------------------------------------------------------
    stage3_group = parser.add_argument_group(
        "Stage 3: per-sample hierarchical integration"
    )

    stage3_group.add_argument(
        "--stage3_prefix",
        type=str,
        default="stage3_integrated",
        help="Output prefix for Stage 3 integrated analysis.",
    )

    stage3_group.add_argument(
        "--stage3_batch_key",
        type=str,
        default="sample",
        help="Batch key used for Harmony integration in Stage 3.",
    )

    stage3_group.add_argument(
        "--stage3_n_pcs",
        type=int,
        default=50,
        help="Number of PCs used for global Stage 3 integration.",
    )

    stage3_group.add_argument(
        "--stage3_n_neighbors",
        type=int,
        default=15,
        help="Number of neighbors for global Stage 3 graph construction.",
    )

    stage3_group.add_argument(
        "--stage3_resolution",
        type=float,
        default=1.0,
        help="Leiden resolution for global Stage 3 integrated clustering.",
    )

    stage3_group.add_argument(
        "--stage3_harmony_theta",
        type=float,
        default=None,
        help=(
            "Optional Harmony theta parameter for Stage 3 integration. "
            "If None, Harmony default is used."
        ),
    )

    # ------------------------------------------------------------------
    # Parse and validate
    # ------------------------------------------------------------------
    parsed_args = parser.parse_args()

    # Manual validation because --samples requires the Stage 3 flag.
    if not (
        parsed_args.data_dir
        or parsed_args.multi_sample
        or (parsed_args.samples and parsed_args.enable_stage3_integration)
    ):
        parser.error(
            "Must specify --data_dir, --multi_sample, or "
            "--samples NAME=PATH ... --enable_stage3_integration"
        )

    if parsed_args.samples and not parsed_args.enable_stage3_integration:
        parser.error(
            "--samples requires --enable_stage3_integration. "
            "Example: --samples WT=/path/wt HD=/path/hd --enable_stage3_integration"
        )

    if parsed_args.enable_stage3_integration and not parsed_args.samples:
        parser.error(
            "--enable_stage3_integration requires --samples NAME=PATH ..."
        )

    # Basic validation of NAME=PATH sample syntax.
    if parsed_args.samples:
        malformed = [s for s in parsed_args.samples if "=" not in s]
        if malformed:
            parser.error(
                "Each --samples entry must use NAME=PATH format. "
                f"Malformed entries: {malformed}"
            )

    # Keep previous behavior for two-sample Harmony/early integration outputs.
    if parsed_args.multi_sample and "harmony" not in parsed_args.output_prefix:
        parsed_args.output_prefix += "_harmony"

    main(parsed_args)


if __name__ == "__main__":
    run()