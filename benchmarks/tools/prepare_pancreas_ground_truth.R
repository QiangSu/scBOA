#!/usr/bin/env Rscript

# ============================================================
# Build an independent non-blood tissue benchmark dataset
# with externally curated cell-type annotations.
#
# Default dataset:
#   Baron et al. human pancreas scRNA-seq from Bioconductor scRNAseq
#
# This script does NOT use Seurat or SeuratData.
# It exports RNA counts in 10x-compatible Matrix Market format
# and saves a metadata CSV containing externally defined labels.
#
# Example:
# Rscript prepare_pancreas_ground_truth.R \
#   --dataset baron_human_pancreas \
#   --output_dir /home/data/qs/scRNA_simulation_data/baron_pancreas_ground_truth \
#   --prefix BaronPancreas \
#   --truth_col label
# ============================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(scRNAseq)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
})

# -----------------------------
# Command-line arguments
# -----------------------------
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[idx + 1]
}

has_flag <- function(flag) {
  flag %in% args
}

if (has_flag("--help") || has_flag("-h")) {
  cat("
Usage:
  Rscript prepare_pancreas_ground_truth.R \\
    --dataset baron_human_pancreas \\
    --output_dir /path/to/output \\
    --prefix BaronPancreas \\
    [--tissue Heart] \\
    [--target_ncells 5000] \\
    [--seed 123] \\
    [--truth_col label]

Arguments:
  --dataset         Dataset name.
                    Supported:
                      baron_human_pancreas
                      muraro_human_pancreas
                      segerstolpe_human_pancreas
                      zeisel_mouse_brain
                      he_organ_atlas

                    Default: baron_human_pancreas

  --tissue          For --dataset he_organ_atlas only. Subset the pooled
                    organ atlas to a single tissue before export.
                    Examples: Heart, Liver, Kidney, Lung, Stomach,
                    Esophagus, Trachea, Muscle, Skin, Bladder,
                    Spleen, Marrow, Common.bile.duct, Rectum, Uterus.
                    If omitted with he_organ_atlas, all available
                    tissues will be listed and the script will exit.

  --output_dir      Output directory.
                    Default: pancreas_ground_truth

  --prefix          Prefix used in summary text.
                    Default: BaronPancreas

  --target_ncells   Optional number of cells to subsample.
                    Default: use all cells

  --seed            Random seed for optional subsampling.
                    Default: 123

  --truth_col       Metadata column to use as external label.
                    If not provided, the script searches common label columns.

Recommended:
  --dataset baron_human_pancreas --truth_col label
  --dataset he_organ_atlas --tissue Heart --truth_col Cell_type
\n")
  quit(save = "no", status = 0)
}

dataset <- get_arg("--dataset", "baron_human_pancreas")
output_dir <- get_arg("--output_dir", "pancreas_ground_truth")
prefix <- get_arg("--prefix", "BaronPancreas")
seed <- as.integer(get_arg("--seed", "123"))

target_ncells_arg <- get_arg("--target_ncells", NA)
target_ncells <- if (is.na(target_ncells_arg)) {
  NULL
} else {
  as.integer(target_ncells_arg)
}

truth_col_arg <- get_arg("--truth_col", NA)
truth_col_arg <- if (is.na(truth_col_arg)) NULL else truth_col_arg

tissue_arg <- get_arg("--tissue", NA)
tissue_arg <- if (is.na(tissue_arg)) NULL else tissue_arg

dir_10x <- file.path(output_dir, "10x_matrix")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dir_10x, recursive = TRUE, showWarnings = FALSE)

set.seed(seed)

if (dataset == "he_organ_atlas" && !is.null(tissue_arg)) {
  prefix <- paste0(prefix, "_", tissue_arg)
}

message("Dataset: ", dataset)
if (!is.null(tissue_arg)) message("Tissue subset: ", tissue_arg)
message("Output directory: ", output_dir)
message("10x matrix directory: ", dir_10x)
message("Prefix: ", prefix)

# -----------------------------
# Helper function for gzip
# -----------------------------
gzip_file <- function(path) {
  if (!file.exists(path)) {
    stop("Cannot gzip file because it does not exist: ", path)
  }

  gz_path <- paste0(path, ".gz")

  if (file.exists(gz_path)) {
    unlink(gz_path)
  }

  status <- system2("gzip", args = c("-f", path))

  if (!file.exists(gz_path)) {
    stop("gzip failed for file: ", path)
  }

  return(gz_path)
}

# -----------------------------
# Load SingleCellExperiment dataset
# -----------------------------
message("Loading dataset from Bioconductor scRNAseq...")

sce <- switch(
  dataset,

  baron_human_pancreas = {
    scRNAseq::BaronPancreasData(which = "human")
  },

  muraro_human_pancreas = {
    scRNAseq::MuraroPancreasData()
  },

  segerstolpe_human_pancreas = {
    scRNAseq::SegerstolpePancreasData()
  },

  zeisel_mouse_brain = {
    scRNAseq::ZeiselBrainData()
  },

  he_organ_atlas = {
    message("Loading HeOrganAtlasData (this may take several minutes on first run)...")
    sce_full <- scRNAseq::HeOrganAtlasData()

    if (!"Tissue" %in% colnames(SummarizedExperiment::colData(sce_full))) {
      stop("HeOrganAtlasData object does not contain a 'Tissue' column in colData.")
    }

    tissues_available <- sort(unique(as.character(sce_full$Tissue)))
    message("Tissues available in HeOrganAtlasData: ",
            paste(tissues_available, collapse = ", "))

    if (is.null(tissue_arg)) {
      cat("\n--- Cell counts per tissue ---\n")
      print(sort(table(sce_full$Tissue), decreasing = TRUE))
      stop(
        "--dataset he_organ_atlas requires --tissue <name>.\n",
        "Please pick one of the tissues listed above and rerun."
      )
    }

    if (!(tissue_arg %in% tissues_available)) {
      stop(
        "Requested --tissue '", tissue_arg, "' not found in HeOrganAtlasData.\n",
        "Available tissues: ", paste(tissues_available, collapse = ", ")
      )
    }

    message("Subsetting to tissue: ", tissue_arg)
    sce_sub <- sce_full[, sce_full$Tissue == tissue_arg]
    message("Retained ", ncol(sce_sub), " cells in tissue '", tissue_arg, "'.")

    rm(sce_full)
    gc(verbose = FALSE)
    sce_sub
  },

  stop(
    "Unsupported --dataset: ", dataset, "\n",
    "Supported datasets are: baron_human_pancreas, muraro_human_pancreas, ",
    "segerstolpe_human_pancreas, zeisel_mouse_brain, he_organ_atlas."
  )
)

message("Loaded SingleCellExperiment with ", ncol(sce), " cells and ", nrow(sce), " features.")
message("Available assays: ", paste(SummarizedExperiment::assayNames(sce), collapse = ", "))
message("Available colData columns: ", paste(colnames(SummarizedExperiment::colData(sce)), collapse = ", "))

# -----------------------------
# Ensure unique cell and gene names
# -----------------------------
if (is.null(colnames(sce))) {
  colnames(sce) <- paste0("Cell_", seq_len(ncol(sce)))
}
colnames(sce) <- make.unique(as.character(colnames(sce)))

if (is.null(rownames(sce))) {
  rownames(sce) <- paste0("Gene_", seq_len(nrow(sce)))
}
rownames(sce) <- make.unique(as.character(rownames(sce)))

# -----------------------------
# Select external annotation column
# -----------------------------
metadata_cols <- colnames(SummarizedExperiment::colData(sce))

truth_candidates <- c(
  "label",
  "celltype",
  "cell_type",
  "Cell_type",
  "Cell_type_in_each_tissue",
  "Cell_type_in_merged_data",
  "cell.type",
  "cell_ontology_class",
  "CellType",
  "cellType",
  "annotation",
  "annotations",
  "assigned_cluster",
  "cluster",
  "level1class",
  "level2class",
  "level3class",
  "primary_type",
  "secondary_type"
)

if (!is.null(truth_col_arg)) {
  if (!truth_col_arg %in% metadata_cols) {
    stop(
      "Requested --truth_col '", truth_col_arg, "' was not found in colData.\n",
      "Available columns: ", paste(metadata_cols, collapse = ", ")
    )
  }
  truth_col <- truth_col_arg
} else {
  matched_cols <- truth_candidates[truth_candidates %in% metadata_cols]
  if (length(matched_cols) == 0) {
    stop(
      "No suitable annotation column found automatically.\n",
      "Available colData columns: ", paste(metadata_cols, collapse = ", "), "\n",
      "Please rerun with --truth_col <column_name>."
    )
  }
  truth_col <- matched_cols[1]
}

message("Using external annotation column: ", truth_col)

truth_labels <- as.character(SummarizedExperiment::colData(sce)[[truth_col]])
names(truth_labels) <- colnames(sce)

missing_labels <- is.na(truth_labels) | truth_labels == ""
if (any(missing_labels)) {
  message("Removing ", sum(missing_labels), " cells with missing labels.")
  sce <- sce[, !missing_labels]
  truth_labels <- truth_labels[!missing_labels]
}

# -----------------------------
# Optional subsampling
# -----------------------------
if (!is.null(target_ncells)) {
  if (target_ncells <= 0) {
    stop("--target_ncells must be a positive integer.")
  }

  if (target_ncells < ncol(sce)) {
    message("Subsampling to ", target_ncells, " cells using seed ", seed, ".")
    selected_cells <- sample(colnames(sce), target_ncells)
    sce <- sce[, selected_cells]
    truth_labels <- truth_labels[selected_cells]
  } else {
    message(
      "--target_ncells is >= available cell number; using all ",
      ncol(sce), " cells."
    )
  }
}

# -----------------------------
# Extract counts
# -----------------------------
message("Extracting count matrix...")

assay_names <- SummarizedExperiment::assayNames(sce)

if ("counts" %in% assay_names) {
  assay_to_use <- "counts"
} else {
  assay_to_use <- assay_names[1]
  message("No assay named 'counts'. Using first assay: ", assay_to_use)
}

counts <- SummarizedExperiment::assay(sce, assay_to_use)

if (!inherits(counts, "dgCMatrix")) {
  counts <- as(as.matrix(counts), "dgCMatrix")
}

message("Count matrix: ", nrow(counts), " genes x ", ncol(counts), " cells.")

if (nrow(counts) == 0 || ncol(counts) == 0) {
  stop("Count matrix is empty. Cannot write 10x files.")
}

# Ensure dimnames
rownames(counts) <- rownames(sce)
colnames(counts) <- colnames(sce)

# -----------------------------
# Write 10x-compatible files
# -----------------------------
matrix_file_plain <- file.path(dir_10x, "matrix.mtx")
features_file_plain <- file.path(dir_10x, "features.tsv")
barcodes_file_plain <- file.path(dir_10x, "barcodes.tsv")

matrix_file <- paste0(matrix_file_plain, ".gz")
features_file <- paste0(features_file_plain, ".gz")
barcodes_file <- paste0(barcodes_file_plain, ".gz")

message("Writing 10x-compatible matrix files...")

unlink(c(
  matrix_file_plain, features_file_plain, barcodes_file_plain,
  matrix_file, features_file, barcodes_file
), force = TRUE)

Matrix::writeMM(obj = counts, file = matrix_file_plain)
matrix_file <- gzip_file(matrix_file_plain)

# Feature table
gene_ids <- rownames(counts)
gene_names <- gene_ids

rowdata <- as.data.frame(SummarizedExperiment::rowData(sce))

possible_symbol_cols <- c(
  "Symbol", "symbol", "gene_symbol", "gene_name",
  "GeneName", "external_gene_name", "feature_symbol"
)

matched_symbol_cols <- possible_symbol_cols[possible_symbol_cols %in% colnames(rowdata)]

if (length(matched_symbol_cols) > 0) {
  tmp_gene_names <- as.character(rowdata[[matched_symbol_cols[1]]])
  if (length(tmp_gene_names) == length(gene_ids)) {
    tmp_gene_names[is.na(tmp_gene_names) | tmp_gene_names == ""] <- gene_ids[is.na(tmp_gene_names) | tmp_gene_names == ""]
    gene_names <- tmp_gene_names
  }
}

features_df <- data.frame(
  gene_id = gene_ids,
  gene_name = gene_names,
  feature_type = rep("Gene Expression", length(gene_ids)),
  stringsAsFactors = FALSE
)

write.table(
  features_df,
  file = features_file_plain,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = FALSE
)
features_file <- gzip_file(features_file_plain)

writeLines(colnames(counts), con = barcodes_file_plain)
barcodes_file <- gzip_file(barcodes_file_plain)

# -----------------------------
# Validate matrix.mtx.gz
# -----------------------------
message("Validating matrix.mtx.gz...")

matrix_header <- readLines(gzfile(matrix_file, open = "rt"), n = 1)

if (length(matrix_header) == 0 || !grepl("^%%MatrixMarket", matrix_header)) {
  stop(
    "Invalid matrix.mtx.gz. First line is not Matrix Market header.\n",
    "Observed first line: ", paste(matrix_header, collapse = " "), "\n",
    "Expected first line beginning with: %%MatrixMarket"
  )
}

message("matrix.mtx.gz header OK: ", matrix_header)

matrix_size <- file.info(matrix_file)$size
features_size <- file.info(features_file)$size
barcodes_size <- file.info(barcodes_file)$size

message("matrix.mtx.gz size: ", matrix_size, " bytes")
message("features.tsv.gz size: ", features_size, " bytes")
message("barcodes.tsv.gz size: ", barcodes_size, " bytes")

if (matrix_size < 1000) {
  stop("matrix.mtx.gz is suspiciously small. File size: ", matrix_size, " bytes.")
}

# -----------------------------
# Write external-label metadata
# -----------------------------
metadata_file <- file.path(output_dir, "ground_truth_metadata.csv")

metadata_out <- data.frame(
  cell_id = colnames(counts),
  ground_truth_cell_type = as.character(truth_labels[colnames(counts)]),
  dataset = dataset,
  external_label_column = truth_col,
  stringsAsFactors = FALSE
)

write.csv(metadata_out, metadata_file, row.names = FALSE)

# -----------------------------
# Write summary
# -----------------------------
label_table <- sort(table(metadata_out$ground_truth_cell_type), decreasing = TRUE)

summary_file <- file.path(output_dir, "dataset_summary.txt")

summary_lines <- c(
  paste0("Dataset: ", prefix, " / ", dataset),
  paste0("Source: Bioconductor scRNAseq"),
  paste0("Object class: SingleCellExperiment"),
  paste0("External annotation column: ", truth_col),
  paste0("Number of cells: ", ncol(counts)),
  paste0("Number of genes/features: ", nrow(counts)),
  paste0("Number of external cell types: ", length(label_table)),
  paste0("Missing labels after filtering: ", sum(is.na(metadata_out$ground_truth_cell_type) | metadata_out$ground_truth_cell_type == "")),
  "",
  "External cell-type counts:",
  paste0(names(label_table), "\t", as.integer(label_table)),
  "",
  "Output files:",
  paste0("1. ", matrix_file),
  paste0("2. ", features_file),
  paste0("3. ", barcodes_file),
  paste0("4. ", metadata_file),
  paste0("5. ", summary_file),
  "",
  "Note:",
  "The cell-type labels are externally curated annotations provided by the original study.",
  "They should be described as external labels rather than absolute biological ground truth."
)

writeLines(summary_lines, con = summary_file)

message("")
message("Done.")
message("Number of cells: ", ncol(counts))
message("Number of genes/features: ", nrow(counts))
message("Number of external cell types: ", length(label_table))
message("External cell-type counts:")
print(label_table)

message("")
message("Output files:")
message("1. ", matrix_file)
message("2. ", features_file)
message("3. ", barcodes_file)
message("4. ", metadata_file)
message("5. ", summary_file)