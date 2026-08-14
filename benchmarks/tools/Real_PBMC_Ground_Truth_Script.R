#!/usr/bin/env Rscript

# ============================================================
# Build a REAL Human PBMC dataset with Ground Truth annotations
# 
# This script BYPASSES Bioconductor databases entirely.
# It directly downloads the raw, physical FACS-sorted matrices 
# from the official 10x Genomics servers, mixes them into a 
# realistic 5000-cell PBMC sample, and outputs the 10x format.
# ============================================================

suppressPackageStartupMessages({
    library(DropletUtils)
    library(Matrix)
    library(utils)
})

# Increase download timeout limit for large files
options(timeout = 600)

set.seed(42)
output_dir <- "real_pbmc_ground_truth"
dir_10x <- file.path(output_dir, "10x_matrix")
temp_dir <- file.path(output_dir, "temp_downloads")

dir.create(dir_10x, recursive = TRUE, showWarnings = FALSE)
dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 1. 10x Genomics Official Download Links
# -----------------------------
# These are the direct links to the physical FACS-sorted populations
populations <- list(
    "CD4_T_Helper" = list(url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd4_t_helper/cd4_t_helper_filtered_gene_bc_matrices.tar.gz", prob = 0.35),
    "CD8_Cytotoxic_T" = list(url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/cytotoxic_t/cytotoxic_t_filtered_gene_bc_matrices.tar.gz", prob = 0.25),
    "Monocytes" = list(url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd14_monocytes/cd14_monocytes_filtered_gene_bc_matrices.tar.gz", prob = 0.20),
    "B_cells" = list(url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz", prob = 0.10),
    "NK_cells" = list(url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd56_nk/cd56_nk_filtered_gene_bc_matrices.tar.gz", prob = 0.10)
)

total_cells <- 5000
sce_list <- list()
metadata_list <- list()

message("Downloading direct pure populations from 10x Genomics servers...")

# -----------------------------
# 2. Download, Extract, and Sample
# -----------------------------
for (pop_name in names(populations)) {
    url <- populations[[pop_name]]$url
    n_target <- round(total_cells * populations[[pop_name]]$prob)
    
    tar_file <- file.path(temp_dir, paste0(pop_name, ".tar.gz"))
    ex_dir <- file.path(temp_dir, pop_name)
    
    # Download file if it doesn't exist
    if (!file.exists(tar_file)) {
        message(sprintf("  -> Downloading %s...", pop_name))
        download.file(url, tar_file, quiet = TRUE, mode = "wb")
    } else {
        message(sprintf("  -> %s already downloaded.", pop_name))
    }
    
    # Extract
    untar(tar_file, exdir = ex_dir)
    
    # Find the actual matrix folder inside the extracted files
    mtx_path <- list.files(ex_dir, pattern = "matrix.mtx", recursive = TRUE, full.names = TRUE)[1]
    data_dir <- dirname(mtx_path)
    
    # Read the 10x data
    sce <- read10xCounts(data_dir)
    
    # Randomly select the exact number of required cells
    sampled_indices <- sample(seq_len(ncol(sce)), size = n_target, replace = FALSE)
    sce_sampled <- sce[, sampled_indices]
    
    # Convert to standard matrix
    mat <- as(counts(sce_sampled), "dgCMatrix")
    
    # Use real human Gene Symbols for rows
    gene_symbols <- rowData(sce_sampled)$Symbol
    # Fallback to Ensembl IDs if a gene has no symbol
    gene_symbols[is.na(gene_symbols) | gene_symbols == ""] <- rowData(sce_sampled)$ID[is.na(gene_symbols) | gene_symbols == ""]
    
    rownames(mat) <- make.unique(gene_symbols)
    
    sce_list[[pop_name]] <- mat
    
    # Record Ground Truth metadata
    metadata_list[[pop_name]] <- data.frame(
        ground_truth_cell_type = rep(pop_name, n_target)
    )
    
    message(sprintf("     Sampled %d pure cells.", n_target))
}

# -----------------------------
# 3. Merge and Save Output
# -----------------------------
message("\nMerging populations into a single realistic PBMC sample...")

# Combine matrices side-by-side (All 10x datasets here share the exact same hg19 gene list)
merged_counts <- do.call(cbind, sce_list)

# Give cells standard names (cell_1, cell_2...)
cell_names <- paste0("cell_", seq_len(ncol(merged_counts)))
colnames(merged_counts) <- cell_names

# Combine Metadata
metadata <- do.call(rbind, metadata_list)
metadata$cell_id <- cell_names
metadata <- metadata[, c("cell_id", "ground_truth_cell_type")]

message("Saving output to modern 10x Genomics format (v3)...")
write10xCounts(
    x = merged_counts, 
    path = dir_10x, 
    version = "3", 
    overwrite = TRUE
)

write.csv(metadata, file.path(output_dir, "ground_truth_metadata.csv"), row.names = FALSE, quote = FALSE)

# Clean up the huge temporary download files to save disk space
unlink(temp_dir, recursive = TRUE)

message("\nSuccess! Files generated in '", output_dir, "':")
message("1. ", file.path(dir_10x, "matrix.mtx.gz"))
message("2. ", file.path(dir_10x, "features.tsv.gz"), " (Contains real human genes)")
message("3. ", file.path(dir_10x, "barcodes.tsv.gz"))
message("4. ", file.path(output_dir, "ground_truth_metadata.csv"), " (Contains perfect cell-type annotations)")