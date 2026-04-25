# In scBOA/README.md

# scBOA: scRNA-seq Bayesian Optimization and Analysis

[![PyPI version](https://img.shields.io/pypi/v/scboa.svg)](https://pypi.org/project/scboa/)
[![GitHub Tests](https://github.com/QiangSu/scBOA/actions/workflows/run_test.yml/badge.svg)](https://github.com/QiangSu/scBOA/actions/workflows/run_test.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/QiangSu/scBOA)

**scBOA** is an integrated, two-stage computational pipeline for single-cell RNA sequencing (scRNA-seq) analysis. It automates the discovery of optimal processing parameters using Bayesian Optimization (Stage 1) and then applies these parameters to a comprehensive downstream analysis workflow (Stage 2). The pipeline also features an optional multi-level refinement process (Stage 3/4) to iteratively re-analyze and improve annotations for low-confidence cell clusters.

## Key Features

-   **Automated Parameter Tuning**: Uses Bayesian Optimization to find the best parameters (`n_highly_variable_genes`, `n_pcs`, `n_neighbors`, `resolution`) for clustering and cell type annotation.
-   **Multi-Metric Objective Function**: Optimizes for a balanced score that considers annotation accuracy (CAS), marker gene specificity (MCS), and cluster separation (Silhouette score).
-   **Single & Multi-Sample Modes**: Natively supports analysis of a single dataset or the integration of two datasets (e.g., control vs. treated) using Harmony.
-   **Iterative Refinement**: Automatically identifies low-confidence cell clusters and re-runs the entire optimization and analysis pipeline on them to improve annotation granularity and accuracy.
-   **Comprehensive Outputs**: Generates publication-quality plots, detailed metric reports, annotated data objects (`.h5ad`), and summary tables for easy interpretation.

---

## Repository Structure

```
scBOA/
├── .github/workflows/
│   └── run_test.yml         # GitHub Action for automated testing
├── example_data/
│   ├── barcodes.tsv.gz      # Example Cell Ranger output
│   ├── features.tsv.gz      # Example Cell Ranger output
│   └── matrix.mtx.gz        # Example Cell Ranger output
├── test_assets/
│   └── Healthy_COVID19_PBMC.pkl  # Pre-trained CellTypist model for testing/examples
├── .gitignore               # Specifies files for Git to ignore
├── LICENSE                  # Project license (e.g., MIT)
├── README.md                # This documentation file
├── references/              # This marker gene driven cell type annotation
├── requirements.txt         # Exact Python dependencies for reproducibility
└── scBOA.py                 # The main executable Python script

```

---

## Step-by-Step Workflow

### 1. Prerequisites

-   Git installed on your system.
-   Python 3.8 or newer.
-   Access to a Linux-based command line.

### 2. Clone the Repository

```bash
git clone https://github.com/QiangSu/scBOA.git
cd scBOA
```

### 3. Set Up a Python Environment (Recommended)

Using a virtual environment prevents conflicts with other Python projects.

```bash
# Create a new conda environment with Python 3.9
conda create -n scboa_env python=3.9

# Activate the environment
conda activate scboa_env

```

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# To deactivate later, simply run: deactivate
```

### 4. Install Dependencies

The `requirements.txt` file contains the exact library versions for perfect reproducibility.

```bash
pip install -r requirements.txt
```

### 5. Prepare Your Data

-   **scRNA-seq Data**: Ensure your Cell Ranger output (the folder containing `barcodes.tsv.gz`, `features.tsv.gz`, and `matrix.mtx.gz`) is accessible.
-   **CellTypist Model**: Download a pre-trained CellTypist model (`.pkl` file). You can find available models on the official [CellTypist models website](https://www.celltypist.org/models).

### 6. Run the Pipeline

Here is an example command for a single-sample analysis:

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./my_analysis_output/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --cas_aggregation_method leiden \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 
```

Single-sample refinement analysis

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./my_analysis_output/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --cas_aggregation_method leiden \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 \
  --cas_refine_threshold 50 \
  --min_cells_refinement 50 \
  --refinement_depth 3 
```

Single-sample multi-function refinement analysis

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --marker_gene_model non-mitochondrial \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 \
  --reference_marker_db ./references/combined_markers_summary.csv \
  --marker_prior_species Human \
  --marker_prior_organ Blood \
  --use_f1 \
  --f1_db_celltype_col cell_type \
  --f1_db_gene_col marker_genes \
  --f1_groupby_key ctpt_consensus_prediction \
  --n_top_genes 50 \
  --cas_aggregation_method leiden \
  --cas_refine_threshold 50 \
  --min_cells_refinement 50 \
  --refinement_depth 3 \
  --mps_bonus_weight 0 \
  --use_confidence \
  --threads 16
```

Multiple-sample refinement analysis:

```bash
python ./scBOA/scBOA.py \
  --multi_sample ./WT_CellRanger/ ./treated_CellRanger/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Mouse_Whole_Brain.pkl \
  --output_prefix WTTR \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --cas_aggregation_method leiden \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 \
  --cas_refine_threshold 50 \
  --min_cells_refinement 50 \
  --refinement_depth 3 
```

Multiple-sample multi-function subsampling refinement analysis:

```bash
python ./scBOA/scBOA.py \
  --multi_sample ./WT_CellRanger/ ./treated_CellRanger/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Mouse_Whole_Brain.pkl \
  --output_prefix WTTR \
  --integration_method harmony \
  --subsample_size 10000 \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --marker_gene_model non-mitochondrial \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 \
  --reference_marker_db ./references/combined_markers_summary.csv \
  --marker_prior_species Mouse \
  --marker_prior_organ Brain \
  --use_f1 \
  --f1_db_celltype_col cell_type \
  --f1_db_gene_col marker_genes \
  --f1_groupby_key ctpt_consensus_prediction \
  --n_top_genes 50 \
  --cas_aggregation_method leiden \
  --cas_refine_threshold 50 \
  --min_cells_refinement 50 \
  --refinement_depth 3 \
  --mps_bonus_weight 0 \
  --use_confidence \
  --threads 16
```

---

## Command-Line Arguments Explained

#### `Stage 1 & 2: Main I/O and Mode`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--data_dir <path>` | Path to 10x Genomics data. | **(Single-Sample Mode)** Provide the path to the directory containing `matrix.mtx.gz`, etc. |
| `--multi_sample <path1> <path2>` | Two paths for WT and Treated 10x data. | **(Multi-Sample Mode)** Provide two paths, first for control/WT, second for treated/perturbed. This mode enables Harmony integration. |
| `--output_dir <path>` | Path for all output files. | The main directory where all results, plots, and logs will be saved. Subdirectories for each stage will be created here. |
| `--integration_method <str>` | Batch correction method. | Default: harmony. Choices: harmony, scanorama, bbknn. Used only when running in --multi_sample mode. |
| `--model_path <path>` | Path to CellTypist model (`.pkl`). | **Required.** The pre-trained model used for cell type annotation. |
| `--output_prefix <str>` | Base prefix for Stage 1 output files. | Default: `bayesian_opt`. Used for naming optimization reports and plots. |
| `--threads <int>` | Number of CPU threads. | Default: 16. Number of threads used for parallel processing (Scanpy njobs). |

#### `Stage 1: Optimization Parameters`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--seed <int>` | Global random seed for reproducibility. | Default: `42`. Ensures that results are identical if run with the same data and parameters. |
| `--n_calls <int>` | Number of trials for EACH optimization strategy. | Default: `50`. The script runs three strategies (Explore, Exploit, BO-EI), so `50` means a total of 150 optimization steps. |
| `--subsample_size <int>` | (Scalability) Max cells for Stage 1. | (Optional) Subsamples large datasets (e.g., 10000) for rapid BO parameter discovery. Stage 2 will automatically apply the optimal parameters to the full dataset. |
| `--model_type <choice>` | Optimization objective function type. | `biological`: Balances annotation agreement (CAS) and marker specificity (MCS). <br> `structural` (default): Adds cluster separation (Silhouette Score) to the biological metrics for more robust clusters. <br> `silhouette`: Optimizes solely for the best Silhouette Score. |
| `--marker_gene_model <choice>` | Genes to use for MCS calculation. | `all`: All genes are considered. <br> `non-mitochondrial` (default): Excludes mitochondrial genes, which often act as non-specific markers of cell stress. |
| `--target <choice>` | Optimization target metric. | `all` (default): Runs a single, balanced optimization (equivalent to `--model_type`). <br> `weighted_cas`, `simple_cas`, `mcs`: Runs optimization targeting only that specific metric. |
| `--cas_aggregation_method <choice>` | Method for calculating Simple CAS. | `leiden` (default): Averages the purity score of each raw Leiden cluster. Best for assessing technical cluster quality. <br> `consensus`: Merges clusters with the same final cell type label before averaging purity. Best for assessing biological group quality. |

#### `Stage 1 & 2: HVG Selection Method`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--hvg_min_mean <float>` | Min mean for two-step HVG selection. | If set, activates a pre-filtering step on genes based on expression and dispersion before selecting the top `n_hvg`. |
| `--hvg_max_mean <float>` | Max mean for two-step HVG selection. | See above. |
| `--hvg_min_disp <float>` | Min dispersion for two-step HVG selection. | See above. |

#### `Stage 1 & 2: QC & Filtering Parameters`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--min_genes <int>` | Min genes per cell. | Default: `200`. Filters out low-quality cells/empty droplets. |
| `--max_genes <int>` | Max genes per cell. | Default: `7000`. Filters out potential doublets. |
| `--max_pct_mt <float>` | Max mitochondrial percentage. | Default: `10.0`. Filters out stressed or dying cells. |
| `--min_cells <int>` | Min cells per gene. | Default: `3`. Filters out genes with negligible expression. |

#### `Stage 2 & Optional Refinement: Final Run Parameters`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--final_run_prefix <str>` | Prefix for Stage 2 output files. | Default: `sc_analysis_repro`. |
| `--fig_dpi <int>` | Resolution (DPI) for saved figures. | Default: `500`. |
| `--n_pcs_compute <int>` | Number of principal components to compute. | Default: `105`. A higher number allows for a wider search space for the optimal `n_pcs`. |
| `--n_top_genes <int>` | Number of top marker genes to show. | Default: `5`. Affects dot plots, heatmaps, and marker gene tables. |
| `--reference_marker_db <path>` | Path to marker DB (.csv). | (Optional) Path to reference database (e.g., PanglaoDB/CellMarker) used for F1 scoring and manual-style annotation. |
| `--marker_prior_species <str>` | Species filter for Marker. | Default: Human. Filters the database to only use markers for this species. |
| `--marker_prior_organ <str>` | Organ filter for Marker. | Default: Blood. Filters the database to only use markers for this tissue type. |
| `--use_f1` | Enable F1 Additive Bonus. | Flag. If provided, calculates F1 similarity against the reference_marker_db and includes it in Stage 1 optimization. |
| `--mps_bonus_weight <float>` | Weight for the F1 Additive Bonus. | Default: 0.2. Adds a max 20% bonus to the objective score based on F1 performance. |
| `--use_confidence` | Enable Confidence Factor. | Flag. Includes the average CellTypist prediction confidence into the geometric mean calculation of the BO objective function. |
| `--f1_db_celltype_col <str>` | Cell type column name. | (Optional) Column name in the Marker containing cell types (auto-detected if blank). |
| `--f1_db_gene_col <str>` | Gene column name. | (Optional) Column name in the Marker DB containing gene lists (auto-detected if blank). |
| `---f1_groupby_key <str>` | Annotation key for F1 scoring. | Default: ctpt_consensus_prediction. Grouping level used when extracting markers to compare against the DB. |
| `--marker_score_metric <choice>` | Metric for manual annotation. | Default: f1. Choices: f1, jaccard, capture. |
| `--n_degs_for_capture <int>` | DEGs for Marker Capture Score. | Default: 5. Top DEGs used to match against the DB if evaluating capture scores. |
| `--n_degs_for_capture <int>` | DEGs for Marker Capture Score. | Default: 5. Top DEGs used to match against the DB if evaluating capture scores. |
| `--cas_refine_threshold <float>`| CAS threshold to trigger refinement. | **(Optional)** If a cluster's CAS score is below this value (e.g., `90`), its cells are pooled for a new round of optimization and analysis. |
| `--refinement_depth <int>` | Maximum number of refinement iterations. | Default: `1`. If refinement is triggered, this controls how many times the process can repeat on the subsequently failing cells. |
| `--min_cells_refinement <int>` | Min cells required to trigger refinement. | Default: `100`. Prevents the pipeline from attempting refinement if the pool of low-quality cells is too small for valid statistical analysis. |

---

## Output Directory Structure

The script generates a structured output directory. Below is an example structure and an explanation of key files.

```
<output_dir>/
├── stage_1_bayesian_optimization/
│   ├── bayesian_opt_structural_balanced_FINAL_annotated.h5ad
│   ├── bayesian_opt_structural_balanced_FINAL_best_params.txt
│   ├── bayesian_opt_structural_balanced_yield_scores_report.csv
│   ├── bayesian_opt_structural_balanced_optimizer_convergence.png
│   ├── bayesian_opt_structural_balanced_BO-EI_opt_result.skopt
│   ├── ... (other plots and strategy files) ...
│   └── refinement_depth_1/
│       ├── ... (mirrors the structure above, but for the refined subset of cells) ...
│
└── stage_2_final_analysis/
    ├── sc_analysis_repro_final_processed.h5ad
    ├── sc_analysis_repro_final_processed_with_refinement.h5ad
    ├── sc_analysis_repro_all_annotations.csv
    ├── sc_analysis_repro_all_annotations_with_refinement.csv
    ├── sc_analysis_repro_leiden_cluster_annotation_scores.csv
    ├── sc_analysis_repro_consensus_group_annotation_scores.csv
    ├── sc_analysis_repro_combined_cluster_annotation_scores.csv
    ├── sc_analysis_repro_cell_type_journey_summary.csv
    ├── sc_analysis_repro_umap_leiden.png
    ├── sc_analysis_repro_cluster_celltypist_umap.png
    ├── sc_analysis_repro_umap_low_confidence_greyed.png
    ├── ... (many other plots and result files) ...
    └── refinement_depth_1/
        ├── sc_analysis_repro_refinement_depth_1_final_processed.h5ad
        ├── sc_analysis_repro_refinement_depth_1_umap_cumulative_result.png
        └── ... (mirrors Stage 2 structure for the refined subset) ...
```

### Key File Explanations

#### Stage 1: `stage_1_bayesian_optimization/`
-   `*_FINAL_best_params.txt`: A summary of the optimal parameters found and the final performance metrics. **This is the most important summary file.**
-   `*_FINAL_annotated.h5ad`: The AnnData object processed with the best parameters, containing all final annotations from the single best run.
-   `*_yield_scores_report.csv`: A detailed log of every trial from every optimization strategy, including parameters tested and all resulting scores (CAS, MCS, Silhouette).
-   `*_optimizer_convergence.png`: A plot showing how the best score improved over time for each strategy.
-   `*_opt_result.skopt`: A saved state of the optimization process, which can be reloaded.

#### Stage 2: `stage_2_final_analysis/`
-   `*_final_processed.h5ad`: The final, fully annotated AnnData object from the initial Stage 2 run. Contains UMAP coordinates, clustering, and all annotations.
-   `*_final_processed_with_refinement.h5ad`: **(If refinement runs)** The master AnnData object with the final, combined annotations after all refinement levels are complete.
-   `*_all_annotations_with_refinement.csv`: A cell-by-cell table of all annotations, including the final `combined_annotation` column after refinement.
-   `*_cluster_annotation_scores.csv`: Tables detailing the Cell Annotation Score (CAS) for each Leiden cluster and each consensus cell type group.
-   `*_combined_cluster_annotation_scores.csv`: A concatenation of all CAS reports from the initial run and all refinement levels.
-   `*_cell_type_journey_summary.csv`: A wide-format table showing how the cell count and CAS score for each cell type change across refinement stages.
-   `*_umap_low_confidence_greyed.png`: A UMAP plot from the initial run where cells belonging to clusters that failed the CAS threshold are colored grey.
-   `refinement_depth_1/*_umap_cumulative_result.png`: A UMAP plot showing the state of the data *after* a refinement level, with newly-annotated cells colored and any still-failing cells shown in grey.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

