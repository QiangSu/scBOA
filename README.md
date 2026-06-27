# In scBOA/README.md

# scBOA: scRNA-seq Bayesian Optimization and Analysis

[![PyPI version](https://img.shields.io/pypi/v/scboa.svg)](https://pypi.org/project/scboa/)
[![GitHub Tests](https://github.com/QiangSu/scBOA/actions/workflows/run_test.yml/badge.svg)](https://github.com/QiangSu/scBOA/actions/workflows/run_test.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://github.com/QiangSu/scBOA)

**scBOA** is an integrated computational pipeline for single-cell RNA sequencing (scRNA-seq) analysis. It automates parameter discovery using Bayesian Optimization, applies the selected parameters to downstream single-cell analysis, and optionally performs iterative refinement of low-confidence clusters.

The current version supports three major workflows:

1. **Single-sample analysis**: Bayesian optimization followed by final clustering, CellTypist annotation, marker evaluation, and optional low-confidence refinement.
2. **Two-sample early-integration analysis**: joint analysis of two datasets, such as control vs. treated, using Harmony, Scanorama, or BBKNN during optimization and final analysis.
3. **Atlas-scale per-sample hierarchical integration**: independent per-sample optimization and annotation followed by Stage 3 global integration using `--samples` together with `--enable_stage3_integration`. This mode separates local biological discovery from global alignment and is designed for multi-sample or atlas-scale studies.

scBOA also includes optional benchmarking backends for equal-budget comparison with Optuna-TPE and random search, probabilistic soft-CAS scoring based on CellTypist probability matrices, marker-prior F1 scoring, confidence-aware optimization, and confidence-based filtering.

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
├── references/              # Marker gene databases used for marker-prior scoring and manual-style annotation
├── requirements.txt         # Exact Python dependencies for reproducibility
└── scBOA.py                 # The main executable Python script

```

## Key Features

- **Automated Parameter Tuning**: Uses Bayesian Optimization to find the best parameters (`n_highly_variable_genes`, `n_pcs`, `n_neighbors`, `resolution`) for clustering and cell type annotation.

- **Multi-Metric Objective Function**: Optimizes a balanced score based on Cell Annotation Score (CAS), Marker Concordance Score (MCS), and optionally Silhouette score, CellTypist confidence, marker-prior F1 score, and soft-CAS.

- **Optimizer Benchmarking Mode**: Supports equal-budget comparison against alternative optimizers using `--benchmark_optimizer optuna` or `--benchmark_optimizer random`. These baselines use the same search space, preprocessing workflow, objective function, and number of evaluations as scBOA.

- **Single-Sample, Two-Sample, and Multi-Sample Modes**:
  - `--data_dir`: single-sample analysis.
  - `--multi_sample WT_DIR TREATED_DIR`: two-sample early-integration analysis.
  - `--samples NAME=PATH ... --enable_stage3_integration`: atlas-scale per-sample hierarchical integration.

- **Multiple Integration Backends**: Supports Harmony, Scanorama, and BBKNN for two-sample early integration through `--integration_method`.

- **Per-Sample Hierarchical Integration**: In Stage 3 mode, scBOA first optimizes and annotates each sample independently, then merges the per-sample outputs and performs global integration. Harmony is precomputed once on the global PCA space before the Stage 3 Optuna optimization, improving scalability for atlas-scale analyses.

- **Soft-CAS Probability Concordance**: Optional soft-CAS metrics use the full CellTypist probability matrix to evaluate probabilistic agreement between cell-level predictions and cluster-level identities. Soft-CAS can be reported with `--compute_soft_cas` or included in the balanced objective with `--use_soft_cas`.

- **Confidence-Aware Analysis**: CellTypist prediction confidence can be included in the optimization objective with `--use_confidence`, and low-confidence cells can optionally be filtered using `--min_confidence`.

- **Iterative Refinement**: Automatically identifies low-CAS clusters and re-runs optimization and final analysis on those cells to improve annotation granularity.

- **Marker-Prior F1 Scoring**: Uses a reference marker database to compute marker precision, recall, F1, Jaccard, and top marker-based matches. The F1 score can optionally be included as an additive objective bonus using `--use_f1`.

- **Comprehensive Outputs**: Generates publication-quality plots, optimization reports, annotated `.h5ad` objects, per-cell annotation tables, marker concordance reports, refinement summaries, Stage 3 integration outputs, and benchmarking files.

---

## Step-by-Step Workflow

### 1. Prerequisites

-   Git installed on your system.
-   Python 3.10 is recommended. The manuscript/revision environment was tested with Python 3.10.18.
-   Access to a Linux-based command line.

### 2. Clone the Repository

```bash
git clone https://github.com/QiangSu/scBOA.git
cd scBOA
```

### 3. Set Up a Python Environment (Recommended)

Using a virtual environment prevents conflicts with other Python projects.

```bash
# Create a new conda environment with Python 3.10
conda create -n scboa_env python=3.10
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

The `requirements.txt` file specifies Python dependencies used for reproducible execution.

```bash
pip install -r requirements.txt
```

### 5. Alternative: Run via Docker (Zero-Installation)

If you prefer not to install Python or Conda environments, you can use the pre-built Docker image. It comes with all dependencies pre-installed and allows you to run `scboa` from any directory.

```bash
docker pull qiangsu/scboa:latest
```

### 6. Prepare Your Data

-   **scRNA-seq Data**: Ensure your Cell Ranger output (the folder containing `barcodes.tsv.gz`, `features.tsv.gz`, and `matrix.mtx.gz`) is accessible.
-   **CellTypist Model**: Download a pre-trained CellTypist model (`.pkl` file). You can find available models on the official [CellTypist models website](https://www.celltypist.org/models).

### 7. Run the Pipeline

At least one input mode must be specified:

- `--data_dir` for single-sample analysis;
- `--multi_sample WT_DIR TREATED_DIR` for two-sample early integration;
- `--samples NAME=PATH ... --enable_stage3_integration` for per-sample hierarchical integration.

If `--sample_names` and `--sample_paths` are used, they must be provided together and have the same length.

Here is an example command for a single-sample analysis:

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./test_assets/Healthy_COVID19_PBMC.pkl \
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
  --model_path ./test_assets/Healthy_COVID19_PBMC.pkl \
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
  --model_path ./test_assets/Healthy_COVID19_PBMC.pkl \
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

Two-sample early-integration refinement analysis:

```bash
python scBOA.py \
  --multi_sample ./WT_CellRanger/ ./treated_CellRanger/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./test_assets/Mouse_Whole_Brain.pkl \
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

Two-sample early-integration multi-function subsampling refinement analysis:

```bash
python scBOA.py \
  --multi_sample ./WT_CellRanger/ ./treated_CellRanger/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./test_assets/Mouse_Whole_Brain.pkl \
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

### 8. Optimizer Benchmarking: scBOA vs Optuna-TPE or Random Search

scBOA includes benchmarking modes for equal-budget comparison with alternative hyperparameter search strategies. These modes use the same preprocessing workflow, search space, objective function, CellTypist model, and evaluation budget as the default scBOA optimizer.

#### Equal-budget Optuna-TPE benchmark

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./benchmark_optuna_output/ \
  --model_path ./test_assets/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample_optuna \
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
  --mps_bonus_weight 0 \
  --use_confidence \
  --benchmark_optimizer optuna \
  --threads 16
```

#### Equal-budget random-search benchmark

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./benchmark_random_output/ \
  --model_path ./test_assets/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample_random \
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
  --mps_bonus_weight 0 \
  --use_confidence \
  --benchmark_optimizer random \
  --threads 16
```

## 9. Atlas-Scale Per-Sample Hierarchical Integration

For datasets with more than two samples, or for atlas-scale analysis where each sample should first be optimized independently, scBOA provides a Stage 3 hierarchical integration mode. This mode is activated with `--samples NAME=PATH ... --enable_stage3_integration`.

In this workflow:

1. Each sample is analyzed independently using the single-sample Stage 1 and Stage 2 workflow.
2. Per-sample optimized annotations and confidence scores are preserved.
3. The per-sample AnnData objects are merged on common genes.
4. A global integration feature space is constructed.
5. Harmony integration is computed once on the global PCA space.
6. Stage 3 Optuna optimization searches over global integration parameters, including `n_pcs`, `n_neighbors`, and Leiden `resolution`.
7. Global clusters are annotated by majority vote from the per-sample optimized annotations.
8. Marker database matching is optionally performed on the integrated labels.

Example command:

```bash
python scBOA.py \
  --samples WT=/path/to/WT_CellRanger/ HD=/path/to/HD_CellRanger/ Replicate3=/path/to/Replicate3_CellRanger/ \
  --enable_stage3_integration \
  --output_dir ./stage3_atlas_output/ \
  --model_path ./test_assets/Mouse_Whole_Brain.pkl \
  --output_prefix atlas \
  --final_run_prefix atlas_final \
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
  --mps_bonus_weight 0 \
  --use_confidence \
  --integration_hvg_strategy sample_specific_union \
  --integration_fixed_n_hvg 3000 \
  --integration_n_pcs 30 \
  --integration_n_neighbors 15 \
  --min_hvg_sample_recurrence 1 \
  --stage3_marker_topN 50 \
  --threads 16
```
In Stage 3 hierarchical integration, Harmony is precomputed once on the merged global PCA representation before the Optuna search. The Stage 3 optimizer then evaluates different numbers of PCs, neighbors, and Leiden resolutions using this precomputed integrated space. This design avoids repeatedly rerunning Harmony inside each optimization trial and improves scalability for large multi-sample analyses.

## 10. Optional Soft-CAS and Confidence-Aware Analysis

scBOA can compute soft-CAS metrics using the full CellTypist probability matrix. Unlike hard CAS, which compares individual CellTypist labels with cluster consensus labels, soft-CAS evaluates probabilistic agreement between each cell’s full prediction distribution and its cluster-level identity distribution.

To compute and report soft-CAS without using it in the optimization objective:

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./soft_cas_report_output/ \
  --model_path ./test_assets/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample_softcas_report \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --compute_soft_cas \
  --threads 16
```

To include soft-CAS in the balanced optimization objective:

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./soft_cas_objective_output/ \
  --model_path ./test_assets/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample_softcas_objective \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --use_soft_cas \
  --threads 16
```
To filter low-confidence CellTypist assignments during downstream analysis, add:
 --min_confidence 0.5




## 11. Benchmarking Outputs

Benchmarking outputs are written under `stage_1_bayesian_optimization/` and include `*_yield_scores_report.csv`, `*_optimizer_convergence.png`, `*_per_trial_exact_scores.png`, and `*_FINAL_best_params.txt`, with the optimizer strategy recorded as `Optuna_TPE` or `Random`.



---

## Command-Line Arguments Explained

#### `Stage 1 & 2: Main I/O and Mode`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--data_dir <path>` | Path to 10x Genomics data. | **(Single-Sample Mode)** Provide the path to the directory containing `matrix.mtx.gz`, etc. |
| `--multi_sample <path1> <path2>` | Two paths for WT and Treated 10x data. | **(Multi-Sample Mode)** Provide two paths, first for control/WT, second for treated/perturbed. This mode enables early two-sample integration using the method selected by `--integration_method`. |
| `--output_dir <path>` | Path for all output files. | The main directory where all results, plots, and logs will be saved. Subdirectories for each stage will be created here. |
| `--integration_method <str>` | Batch correction method. | Default: harmony. Choices: harmony, scanorama, bbknn. Used only when running in --multi_sample mode. |
| `--model_path <path>` | Path to CellTypist model (`.pkl`). | **Required.** The pre-trained model used for cell type annotation. |
| `--output_prefix <str>` | Base prefix for Stage 1 output files. | Default: `bayesian_opt`. Used for naming optimization reports and plots. |
| `--threads <int>` | Number of CPU threads. | Default: 16. Number of threads used for parallel processing (Scanpy njobs). |
| `--benchmark_optimizer <choice>` | Optional optimizer backend for benchmarking. | Choices: `gp`, `optuna`, `random`. Default: `None`, equivalent to the default scBOA GP-BO backend. `optuna` runs an equal-budget Optuna-TPE backend. `random` runs an equal-budget random-search baseline. Both use the same search space, objective function, and number of evaluations as the selected scBOA run. |

#### `Stage 1: Optimization Parameters`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--seed <int>` | Global random seed for reproducibility. | Default: `42`. Ensures that results are identical if run with the same data and parameters. |
| `--n_calls <int>` | Number of optimization trials. | Default: `50`. For the default GP-BO backend, this value is used for each internal BO strategy where applicable. For `--benchmark_optimizer optuna` or `--benchmark_optimizer random`, this is the equal-budget number of benchmark trials. |
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
| `--f1_db_celltype_col <str>` | Cell type column name. | (Optional) Column name in the Marker containing cell types (auto-detected if blank). |
| `--f1_db_gene_col <str>` | Gene column name. | (Optional) Column name in the Marker DB containing gene lists (auto-detected if blank). |
| `--f1_groupby_key <str>` | Annotation key for F1 scoring. | Default: ctpt_consensus_prediction. Grouping level used when extracting markers to compare against the DB. |
| `--marker_score_metric <choice>` | Metric for manual annotation. | Default: f1. Choices: f1, jaccard, capture. |
| `--n_degs_for_capture <int>` | DEGs for Marker Capture Score. | Default: 5. Top DEGs used to match against the DB if evaluating capture scores. |
| `--cas_refine_threshold <float>`| CAS threshold to trigger refinement. | **(Optional)** If a cluster's CAS score is below this value (e.g., `90`), its cells are pooled for a new round of optimization and analysis. |
| `--refinement_depth <int>` | Maximum number of refinement iterations. | Default: `1`. If refinement is triggered, this controls how many times the process can repeat on the subsequently failing cells. |
| `--min_cells_refinement <int>` | Min cells required to trigger refinement. | Default: `100`. Prevents the pipeline from attempting refinement if the pool of low-quality cells is too small for valid statistical analysis. |
| `--use_confidence` | Include CellTypist confidence in objective. | Flag. If provided, the mean CellTypist confidence is included as an additional factor in the balanced geometric-mean objective. |
| `--min_confidence <float>` | Filter low-confidence cells. | Optional threshold in `[0,1]`. Cells with `ctpt_confidence` below this value are removed from downstream analysis. |
| `--compute_soft_cas` | Compute soft-CAS metrics. | Flag. Computes probability-based soft-CAS metrics from the full CellTypist probability matrix. Does not affect optimization unless `--use_soft_cas` is also set. |
| `--use_soft_cas` | Include soft-CAS in balanced objective. | Flag. Automatically enables `--compute_soft_cas`. Adds the soft-CAS dot-product score as an additional factor in the balanced objective. |

#### `Stage 3: Multi-Sample Hierarchical Integration`

| Argument | Description | Explanation/Usage |
| :--- | :--- | :--- |
| `--samples NAME=PATH ...` | Multiple sample inputs. | Provides two or more samples as `NAME=PATH` pairs. Each sample is independently optimized and analyzed before global integration. Must be used with `--enable_stage3_integration`. |
| `--sample_names <names...>` | Alternative sample-name input. | Space-separated sample names. Must be paired 1:1 with `--sample_paths`. These are internally converted to `--samples NAME=PATH` format. |
| `--sample_paths <paths...>` | Alternative sample-path input. | Space-separated sample directories. Must be paired 1:1 with `--sample_names`. |
| `--enable_stage3_integration` | Enable Stage 3 integration. | Flag. Activates the per-sample hierarchical integration pipeline. Required when using `--samples`, `--sample_names`, or `--sample_paths` as the main input mode. |
| `--integration_hvg_strategy <choice>` | HVG strategy for Stage 3 integration. | Choices: `fixed_global`, `batch_consensus`, `sample_specific_union`, `sample_specific_weighted`. Default: `sample_specific_union`. |
| `--integration_fixed_n_hvg <int>` | Number of HVGs for fixed/global Stage 3 strategies. | Default: `3000`. Used for `fixed_global`, `batch_consensus`, or fallback HVG selection. |
| `--integration_n_pcs <int>` | Number of PCs for the standard Stage 3 integration anchor. | Default: `30`. Used for the original/heuristic integration comparison. |
| `--integration_n_neighbors <int>` | Number of neighbors for the standard Stage 3 integration anchor. | Default: `15`. |
| `--min_hvg_sample_recurrence <int>` | Minimum HVG recurrence across samples. | Default: `1`. For sample-specific union/weighted strategies, a gene must be selected as HVG in at least this many samples to enter the integration feature space. |
| `--integration_max_union_genes <int>` | Maximum union genes for integration. | Default: `5000`. Reserved for controlling the maximum size of the integration feature space. |
| `--stage3_marker_topN <int>` | Top DE genes for Stage 3 marker matching. | Default: `50`. Number of top non-mitochondrial DE genes per integrated label used for marker database F1 matching. |


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
│   └── refinement_depth_1/
│       └── ... 
├── stage_2_final_analysis/
│   ├── sc_analysis_repro_final_processed.h5ad
│   ├── sc_analysis_repro_final_processed_with_refinement.h5ad
│   ├── sc_analysis_repro_all_annotations.csv
│   ├── sc_analysis_repro_all_annotations_with_refinement.csv
│   ├── sc_analysis_repro_leiden_cluster_annotation_scores.csv
│   ├── sc_analysis_repro_consensus_group_annotation_scores.csv
│   ├── sc_analysis_repro_combined_cluster_annotation_scores.csv
│   ├── sc_analysis_repro_cell_type_journey_summary.csv
│   ├── sc_analysis_repro_umap_leiden.png
│   ├── sc_analysis_repro_cluster_celltypist_umap.png
│   ├── sc_analysis_repro_umap_low_confidence_greyed.png
│   └── refinement_depth_1/
│       ├── sc_analysis_repro_refinement_depth_1_final_processed.h5ad
│       ├── sc_analysis_repro_refinement_depth_1_umap_cumulative_result.png
│       └── ...
└── stage_3_combined_integration/
    ├── <prefix>_stage3_merged_premerge.h5ad
    ├── <prefix>_stage3_PRE_batch_umap.png
    ├── <prefix>_stage3_PRE_celltypes_umap.png
    ├── <prefix>_stage3_ORIGINAL_POST_batch_umap.png
    ├── <prefix>_stage3_ORIGINAL_POST_celltypes_umap.png
    ├── <prefix>_stage3_OPTIMIZED_POST_batch_umap.png
    ├── <prefix>_stage3_OPTIMIZED_POST_Global_Majority_Vote_umap.png
    ├── <prefix>_stage3_OPTIMIZED_POST_Leiden_Clusters_umap.png
    ├── <prefix>_stage3_OPTIMIZED_POST_markerCelltypes_umap.png
    ├── <prefix>_stage3_OPTIMIZED_marker_celltype_mapping.csv
    ├── <prefix>_stage3_integration_features.csv
    ├── <prefix>_stage3_gene_weights.csv
    ├── <prefix>_stage3_annotations.csv
    └── <prefix>_stage3_COMBINED_integrated.h5ad
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
#### Stage 3: `stage_3_combined_integration/`
-   `*_merged_premerge.h5ad`: merged object before final Stage 3 integration.
-   `*_PRE_batch_umap.png`: pre-integration UMAP colored by batch/sample.
-   `*_PRE_celltypes_umap.png`: pre-integration UMAP colored by per-sample CellTypist labels.
-   `*_ORIGINAL_POST_*_umap.png`: standard heuristic integration outputs.
-   `*_OPTIMIZED_POST_*_umap.png`: optimized Stage 3 integration outputs.
-   `*_OPTIMIZED_marker_celltype_mapping.csv`: marker database matching results for optimized global labels.
-   `*_integration_features.csv`: genes used for Stage 3 integration.
-   `*_gene_weights.csv`: gene weights, when using sample_specific_weighted.
-   `*_annotations.csv`: per-cell Stage 3 annotations.
-   `*_COMBINED_integrated.h5ad`: final integrated AnnData object.

---

## End-to-End Reproducibility Tutorial (Human PBMC)

To ensure complete transparency and reproducibility, this tutorial provides a step-by-step guide to recreating the exact Human PBMC analysis presented in the scBOA manuscript. By following these steps, users can verify the pipeline’s performance from raw reads to final visualizations using publicly available 10x Genomics data.

### Step 1: Data Acquisition and Read Alignment
We utilize the publicly available **10k Human PBMC dataset** (10x Genomics). The reference genome is derived from the standard `refdata-gex-GRCh38-2024-A` package. Processed reads were aligned using Cell Ranger (v9.0.1) to generate the initial cell-gene matrix. 

*If you already have your own Cell Ranger output, you can skip to Step 2.*

```bash
cellranger count \
  --id=pbmc_10k_v3_output \
  --sample=pbmc_10k_v3 \
  --fastqs=./scRNA_seq_data_public/pbmc_10k_v3_fastqs/cellranger_output \
  --transcriptome=./tools/refdata-gex-GRCh38-2024-A \
  --localcores=40 \
  --localmem=200 \
  --expect-cells=12000 \
  --create-bam=true
```

### Step 2: Runnable scBOA Execution
Using the output from Cell Ranger, execute the complete scBOA pipeline using the following single command. This command incorporates the identical parameters used in our manuscript, including Bayesian Optimization (50 calls), multi-stage refinement (depth 3), and integration of marker gene F1-scoring.
(Ensure you have downloaded the Healthy_COVID19_PBMC.pkl model and have the combined_markers_summary.csv file from the references/ folder).

You can execute this via Local Python or Docker.

#### Option A: Execution via Local Python

(Assuming you are inside the cloned scBOA repository)

```bash
python ./scBOA.py \
  --data_dir ./pbmc_10k_v3_output/outs/filtered_feature_bc_matrix_biological \
  --output_dir ./10kPBMC_BOCR_biological_leiden_50calls_50cas_r3/ \
  --model_path ./test_assets/Healthy_COVID19_PBMC.pkl \
  --output_prefix 10kPBMC \
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
  --f1_db_celltype_col cell_type \
  --f1_db_gene_col marker_genes \
  --cas_aggregation_method leiden \
  --cas_refine_threshold 50 \
  --min_cells_refinement 50 \
  --refinement_depth 3 \
  --threads 90 \
  --use_f1 \
  --n_top_genes 50 \
  --f1_groupby_key ctpt_consensus_prediction \
  --mps_bonus_weight 0 \
  --use_confidence
```
#### Option B: Execution via Docker (Zero-Installation)

Docker allows you to run the pipeline without installing any Python dependencies. By mounting your host directory using `-v` and passing your local user permissions, the container will seamlessly read your data and write the outputs directly back to your computer.

*(Note: The `-v /home/data:/home/data` flag ensures the container uses the exact same absolute file paths as your local machine, meaning you don't have to change any paths in the command).*

```bash
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -w /tmp \
  -v /home/data:/home/data \
  qiangsu/scboa:latest scboa \
  --data_dir ./cellranger_output/pbmc_10k_v3_output/outs/filtered_feature_bc_matrix_biological \
  --output_dir /home/data/10kPBMC_BOCR_biological_docker_output1/ \
  --model_path /home/data/.celltypist/data/reference/Healthy_COVID19_PBMC.pkl \
  --output_prefix 10kPBMC \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --marker_gene_model non-mitochondrial \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 \
  --reference_marker_db /home/data/references/combined_markers_summary.csv \
  --marker_prior_species Human \
  --marker_prior_organ Blood \
  --f1_db_celltype_col cell_type \
  --f1_db_gene_col marker_genes \
  --cas_aggregation_method leiden \
  --cas_refine_threshold 50 \
  --min_cells_refinement 50 \
  --refinement_depth 3 \
  --threads 90 \
  --use_f1 \
  --n_top_genes 50 \
  --f1_groupby_key ctpt_consensus_prediction \
  --mps_bonus_weight 0 \
  --use_confidence
```

### Step 3: Interpretable, Ready-to-Use Outputs
scBOA is designed to automatically generate a highly organized directory tree. Running the command above generates a structured output directory containing publication-ready `.png` visualizations, `.csv` summary tables, optimized parameter reports, and annotated `.h5ad` files. 

To provide immediate proof of reproducibility, we have uploaded this exact output directory (10kPBMC_BOCR_biological_leiden_50calls_50cas_r3/) directly to our GitHub repository. 

Users can browse this folder on GitHub to instantly view the iterative refinement UMAPs, marker concordance scores, and parameter convergence plots. The high-level structure of this output is organized as follows:

```
10kPBMC_BOCR_biological_leiden_50calls_50cas_r3/
├── stage_1_bayesian_optimization/
│   ├── 10kPBMC_biological_balanced_optimizer_convergence.png
│   ├── 10kPBMC_biological_balanced_yield_scores_report.csv
│   ├── refinement_depth_1/   # (Corresponding BO plots for iteration 1)
│   ├── refinement_depth_2/   # (Corresponding BO plots for iteration 2)
│   └── refinement_depth_3/   # (Corresponding BO plots for iteration 3)
└── stage_2_final_analysis/
    ├── sc_analysis_repro_cell_type_journey_summary.csv
    ├── sc_analysis_repro_f1_annotation_summary.csv
    ├── sc_analysis_repro_umap_combined_annotation_final.png
    ├── refinement_depth_1/   # (Final analysis UMAPs & F1 scores for iteration 1)
    ├── refinement_depth_2/   # (Final analysis UMAPs & F1 scores for iteration 2)
    └── refinement_depth_3/   # (Final analysis UMAPs & F1 scores for iteration 3)
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

