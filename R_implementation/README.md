# scBOA: Seurat/R Implementation

This directory contains the **R implementation** of scBOA (Single-Cell Bayesian Optimization for Annotation). This version utilizes the **Seurat** ecosystem and **Cross-Dataset Anchoring** to optimize analysis parameters.

It is designed as a standalone script that supports both **Human** and **Mouse** data.

## Dependencies

Ensure you have R installed (tested on R >= 4.0). You will need the following packages:

```r
install.packages(c("Seurat", "dplyr", "ggplot2", "argparse", "rBayesianOptimization", "Rtsne", "BiocManager"))
BiocManager::install(c("AnnotationDbi", "org.Hs.eg.db", "org.Mm.eg.db"))
```

## Usage

The script requires:
*   Query Data: A directory containing 10x Genomics output (barcodes, features, matrix).
*   Reference Data: A pre-annotated Seurat object (.rds) to serve as the "ground truth" for label transfer during optimization.

**Example Command** 
```bash
Rscript scBOA_Seurat.R \
  --data_dir /path/to/10x_data/ \
  --output_dir ./results_single/ \
  --reference_path ./references/pbmc3k_final_reference.rds  \
  --reference_labels_col seurat_annotations \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --subsample_n_cells 5000 \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 \
  --species human \
  --model_type biological

```

## Optimization Models
*   --model_type structural (Default): Optimizes for CAS (consistency), MCS (biological marker specificity), and Silhouette Score (cluster separation).
*   --model_type biological: Optimizes only for CAS and MCS.