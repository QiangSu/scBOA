# scRNA-seq Data Simulation Workflow

This directory contains scripts to generate synthetic single-cell RNA-seq data compatible with the 10x Genomics Cell Ranger pipeline. The workflow simulates biological heterogeneity using **Splatter**, generates raw reads using **Polyester**, and post-processes the data to add valid 10x barcodes, UMIs, and Poly-T tails.

## Overview

The simulation consists of two steps:
1.  **Biological Simulation (R):** Generates a count matrix and simulates raw transcript reads per cell.
2.  **Library Construction Simulation (Python):** Formatting reads into Cell Ranger-compatible FASTQ files (R1/R2) with valid technical sequences.

## Prerequisites

### R Dependencies (Step 1)
*   R version >= 4.0
*   `splatter`
*   `polyester`
*   `Biostrings`
*   `SummarizedExperiment`
*   `Matrix`

### Python Dependencies (Step 2)
*   Python 3.x
*   Standard libraries: `gzip`, `os`, `sys`, `random`

### External Resources
1.  **Reference Transcriptome:** A FASTA file (e.g., `Mus_musculus.GRCm38.cdna.all.fa`).
2.  **10x Barcode Whitelist:** A text file containing valid barcodes (e.g., `3M-february-2018.txt.gz` found in Cell Ranger `lib/python/cellranger/barcodes/`).

---

## Configuration

**IMPORTANT:** Both scripts contain hardcoded paths that must be updated to match your environment before running.

1.  **In `simu_step1_....R`**:
    *   Update `base_dir`: Where output directories will be created.
    *   Update `reference_fasta`: Path to your reference cDNA FASTA.

2.  **In `simu_step2_....py`**:
    *   Update `base_dir`: Must match the directory used in Step 1.
    *   Update `barcode_whitelist_path`: Path to the 10x Genomics whitelist file.

---

## Workflow

### Step 1: Simulate Counts and Raw Reads
This script simulates 15,000 cells across 10 cell types ("groups") with high differential expression signal.

```bash
Rscript simu_step1_simulate_scRNA_seq_data_15000c_10g_standard_2.R

```
Outputs:

ground_truth_matrix.csv: The gene count matrix.
ground_truth_celltypes.txt: Cell ID to Cell Type mapping.
simulated_fastq_15000c_10g_standard/: Directory containing individual FASTA/FASTQ files for each cell.

### Step 2: Add Barcodes, UMIs, and Format for Cell Ranger
This script mimics the 10x Genomics 3' v3 library structure. It samples real barcodes, generates random UMIs, creates Read 1 (Barcode+UMI+PolyT) and Read 2 (Transcript), and merges all cells into a single pair of FASTQ files.

```bash
python3 simu_step2_add_barcodes_umis_cellranger_15000c_10g_standard_2.py
```
Outputs:

simulated_S1_R1_001.fastq.gz: Read 1 (16bp Barcode + 12bp UMI + 20bp PolyT).
simulated_S1_R2_001.fastq.gz: Read 2 (cDNA sequence).
simulated_barcode_mapping.txt: Log file mapping input cell indices to the assigned 10x barcode.

### Downstream Usage
The output FASTQ files can be directly processed using cellranger count. Ensure you use the --sample name defined in the Python script (default: simulated).

Example:
cellranger count --id=simulated_run \
                 --sample=simulated \
                 --fastqs=/path/to/output_dir \
                 --transcriptome=/path/to/refdata \
                 --expect-cells=15000