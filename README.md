# In scBOA/README.md

# scBOA: scRNA-seq Bayesian Optimization and Analysis

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
├── .gitignore               # Specifies files for Git to ignore
├── LICENSE                  # Project license (e.g., MIT)
├── README.md                # This documentation file
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
  --model_path /path/to/your/celltypist_model.pkl \
  --n_calls 50 \
  --target balanced \
  --model_type structural \
  --cas_refine_threshold 90 \
  --refinement_depth 2
```

---

## Command-Line Arguments Explained

#### `Stage 1 & 2: Main I/O and Mode`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--data_dir <path>` | Path to 10x Genomics data. | **(Single-Sample Mode)** Provide the path to the directory containing `matrix.mtx.gz`, etc. |
| `--multi_sample <path1> <path2>` | Two paths for WT and Treated 10x data. | **(Multi-Sample Mode)** Provide two paths, first for control/WT, second for treated/perturbed. This mode enables Harmony integration. |
| `--output_dir <path>` | Path for all output files. | The main directory where all results, plots, and logs will be saved. Subdirectories for each stage will be created here. |
| `--model_path <path>` | Path to CellTypist model (`.pkl`). | **Required.** The pre-trained model used for cell type annotation. |
| `--output_prefix <str>` | Base prefix for Stage 1 output files. | Default: `bayesian_opt`. Used for naming optimization reports and plots. |

#### `Stage 1: Optimization Parameters`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--seed <int>` | Global random seed for reproducibility. | Default: `42`. Ensures that results are identical if run with the same data and parameters. |
| `--n_calls <int>` | Number of trials for EACH optimization strategy. | Default: `50`. The script runs three strategies (Explore, Exploit, BO-EI), so `50` means a total of 150 optimization steps. |
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
| `--cellmarker_db <path>` | Path to a cell marker database (.csv). | **(Optional)** If provided, performs a "manual-style" annotation based on cluster marker genes and calculates a Marker Capture Score. |
| `--n_degs_for_capture <int>` | DEGs per cluster for Marker Capture Score. | Default: `50`. Number of top differentially expressed genes used to match against the marker DB. |
| `--cas_refine_threshold <float>`| CAS threshold to trigger refinement. | **(Optional)** If a cluster's CAS score is below this value (e.g., `90`), its cells are pooled for a new round of optimization and analysis. |
| `--refinement_depth <int>` | Maximum number of refinement iterations. | Default: `1`. If refinement is triggered, this controls how many times the process can repeat on the subsequently failing cells. |

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

