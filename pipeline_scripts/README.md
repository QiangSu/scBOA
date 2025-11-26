# scBOA Analysis Pipelines

This directory contains standalone Scanpy pipelines designed to perform **reproducible, end-to-end analysis** using specific parameter sets (e.g., optimal parameters identified by the scBOA optimization engine).

These scripts ensure that the final biological interpretation (plots, metrics, and annotations) is generated using a fixed random seed and rigorous quality control standards.

## Scripts Overview

### 1. `Scanpy_scBOA_pipeline_single.py`
**Purpose:** Analysis of a single 10x Genomics dataset.
**Key Features:**
*   **Quality Control:** Robust mitochondrial filtering.
*   **Metrics:** Calculates **CAS** (Cluster Annotation Score) and **MCS** (Marker Concordance Score).
*   **Annotation:** Supports CellTypist, manual annotation via marker CSV, and "Smart Ratio" annotation.
*   **Output:** UMAPs, Dotplots, CAS/MCS tables, and a processed `.h5ad` file.

**Usage Example:**
```bash
python Scanpy_scBOA_pipeline_single.py \
    --data_dir /path/to/10x_data/ \
    --output_dir ./results_single/ \
    --celltypist_model /path/to/model.pkl \
    --n_hvgs 3000 --n_pcs 50 --resolution 1.5 \
    --marker_db_csv /path/to/markers.csv
```

### 2. `Scanpy_scBOA_pipeline_multi.py`
**Purpose:** Integrated analysis of two conditions (e.g., Wild-Type vs. Treated).
**Key Features:**
*   **Integration:** Concatenates samples and performs Harmony integration (if installed).
*   **Comparisons:** Performs Compositional Analysis (cell type abundance changes) and Differential Gene Expression (DGE).
*   **Consistency:** Ensures plotting aesthetics match the single-sample pipeline.

**Usage Example:**
```bash
ppython Scanpy_scBOA_pipeline_multi.py \
    --wt_path /path/to/WT_data/ \
    --treated_path /path/to/Treated_data/ \
    --output_dir ./results_integrated/ \
    --n_hvgs 3000 --n_pcs 80 --resolution 2.0

### Dependencies
*   `scanpy`
*   `celltypist`
*   `harmonypy` (for multi-sample integration)
*   `matplotlib`
*   `SummarizedExperiment`
*   `pandas`
*   `numpy`