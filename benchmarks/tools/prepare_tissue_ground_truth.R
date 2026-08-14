#!/usr/bin/env Rscript

# ============================================================
# Build independent tissue benchmark datasets with externally
# curated cell-type annotations, exported as 10x MTX format.
#
# Supported datasets:
#
#   From Bioconductor scRNAseq (loaded via R API):
#     baron_human_pancreas         (Baron 2016,   endocrine)
#     muraro_human_pancreas        (Muraro 2016,  endocrine)
#     segerstolpe_human_pancreas   (Segerstolpe 2016, endocrine)
#     zeisel_mouse_brain           (Zeisel 2015,  mouse cortex)
#
#   From public .h5ad files (downloaded on first use):
#     litvinukova_human_heart      (Litvinukova 2020, HCA Heart Atlas, 10x)
#     macparland_human_liver       (MacParland 2018, human liver, 10x)
#
# This script does NOT use Seurat or SeuratData.
# It exports RNA counts in 10x-compatible Matrix Market format
# and saves a metadata CSV containing externally defined labels.
#
# Examples:
#
# Rscript prepare_tissue_ground_truth.R \
#   --dataset baron_human_pancreas \
#   --output_dir /home/data/qs/scRNA_simulation_data/baron_pancreas_ground_truth \
#   --prefix BaronPancreas \
#   --truth_col label
#
# Rscript prepare_tissue_ground_truth.R \
#   --dataset litvinukova_human_heart \
#   --output_dir /home/data/qs/scRNA_simulation_data/litvinukova_heart_ground_truth \
#   --prefix LitvinukovaHeart \
#   --truth_col cell_type
#
# Rscript prepare_tissue_ground_truth.R \
#   --dataset macparland_human_liver \
#   --output_dir /home/data/qs/scRNA_simulation_data/macparland_liver_ground_truth \
#   --prefix MacParlandLiver \
#   --truth_col CellType
# ============================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(scRNAseq)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  # Needed for .h5ad datasets (heart / liver / kidney).
  # Install once with:
  #   BiocManager::install("zellkonverter")
  library(zellkonverter)
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
  Rscript prepare_tissue_ground_truth.R \\
    --dataset <name> \\
    --output_dir /path/to/output \\
    --prefix <Prefix> \\
    [--target_ncells N] \\
    [--seed 123] \\
    [--truth_col <col>] \\
    [--h5ad_path /path/to/local.h5ad] \\
    [--counts_layer <layer_name>]

Supported --dataset values:
  baron_human_pancreas
  muraro_human_pancreas
  segerstolpe_human_pancreas
  zeisel_mouse_brain
  litvinukova_human_heart       (10x, HCA Heart Cell Atlas)
  macparland_human_liver        (10x, MacParland 2018)

Notes on --counts_layer:
  For .h5ad datasets whose X is log-normalized, pass e.g.
    --counts_layer counts
  to force the loader to use layers['counts'] as the raw UMI matrix.
  If not provided, the loader will pick 'counts' automatically when
  present, else fall back to X and warn if non-integer values are seen.

Recommended:
  --dataset baron_human_pancreas    --truth_col label
  --dataset litvinukova_human_heart --truth_col cell_type
  --dataset macparland_human_liver  --truth_col CellType
\n")
  quit(save = "no", status = 0)
}

dataset <- get_arg("--dataset", "baron_human_pancreas")
output_dir <- get_arg("--output_dir", "tissue_ground_truth")
prefix <- get_arg("--prefix", "Tissue")
seed <- as.integer(get_arg("--seed", "123"))

target_ncells_arg <- get_arg("--target_ncells", NA)
target_ncells <- if (is.na(target_ncells_arg)) NULL else as.integer(target_ncells_arg)

truth_col_arg <- get_arg("--truth_col", NA)
truth_col_arg <- if (is.na(truth_col_arg)) NULL else truth_col_arg

h5ad_path_arg <- get_arg("--h5ad_path", NA)
h5ad_path_arg <- if (is.na(h5ad_path_arg)) NULL else h5ad_path_arg

counts_layer_arg <- get_arg("--counts_layer", NA)
counts_layer_arg <- if (is.na(counts_layer_arg)) NULL else counts_layer_arg

dir_10x <- file.path(output_dir, "10x_matrix")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dir_10x, recursive = TRUE, showWarnings = FALSE)

set.seed(seed)

message("Dataset: ", dataset)
message("Output directory: ", output_dir)
message("10x matrix directory: ", dir_10x)
message("Prefix: ", prefix)

# -----------------------------
# Helper: gzip a plain file in place
# -----------------------------
gzip_file <- function(path) {
  if (!file.exists(path)) {
    stop("Cannot gzip file because it does not exist: ", path)
  }
  gz_path <- paste0(path, ".gz")
  if (file.exists(gz_path)) unlink(gz_path)
  system2("gzip", args = c("-f", path))
  if (!file.exists(gz_path)) stop("gzip failed for file: ", path)
  gz_path
}

# -----------------------------
# Helper: generic .h5ad -> SingleCellExperiment loader
# Downloads the file on first use, caches it in output_dir.
# -----------------------------
load_h5ad_sce <- function(url, output_dir, h5ad_path = NULL,
                          counts_layer = NULL) {
  if (!is.null(h5ad_path) && nzchar(h5ad_path)) {
    local_path <- h5ad_path
  } else {
    local_path <- file.path(output_dir, basename(url))
  }

  if (!file.exists(local_path)) {
    message("Downloading .h5ad from: ", url)
    message("Saving to: ", local_path)
    dir.create(dirname(local_path), recursive = TRUE, showWarnings = FALSE)
    old_timeout <- getOption("timeout")
    options(timeout = 3600)  # up to 1h for multi-GB downloads
    on.exit(options(timeout = old_timeout), add = TRUE)
    utils::download.file(
      url = url,
      destfile = local_path,
      mode = "wb",
      quiet = FALSE
    )
  } else {
    message("Using cached .h5ad: ", local_path)
  }

  message("Reading .h5ad into SingleCellExperiment via zellkonverter...")
  sce <- zellkonverter::readH5AD(local_path, use_hdf5 = FALSE)

  an <- SummarizedExperiment::assayNames(sce)
  message("Available assays in .h5ad: ", paste(an, collapse = ", "))

  # Priority: explicit --counts_layer > existing 'counts' > raw > X
  chosen <- NULL
  if (!is.null(counts_layer) && nzchar(counts_layer)) {
    if (!counts_layer %in% an) {
      stop("Requested --counts_layer '", counts_layer,
           "' not found. Available: ", paste(an, collapse = ", "))
    }
    chosen <- counts_layer
  } else if ("counts" %in% an) {
    chosen <- "counts"
  } else if ("raw" %in% an) {
    chosen <- "raw"
  } else if ("X" %in% an) {
    chosen <- "X"
  } else {
    chosen <- an[1]
  }

  if (chosen != "counts") {
    message("Renaming assay '", chosen, "' -> 'counts' for downstream use.")
    idx <- which(an == chosen)
    SummarizedExperiment::assayNames(sce)[idx] <- "counts"
  }

  sce
}

# -----------------------------
# Load SingleCellExperiment dataset
# -----------------------------
message("Loading dataset...")

sce <- switch(
  dataset,

  # ---- Bioconductor scRNAseq ----
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

  # ---- .h5ad-based datasets ----

  # Litvinukova et al. 2020, Nature. HCA Heart Cell Atlas.
  # 10x + 10x-nuclei, ~490k cells across four heart regions.
  # Source: https://www.heartcellatlas.org/
  litvinukova_human_heart = {
    load_h5ad_sce(
      url = paste0(
        "https://cellgeni.cog.sanger.ac.uk/heartcellatlas/data/",
        "global_raw.h5ad"
      ),
      output_dir = output_dir,
      h5ad_path = h5ad_path_arg,
      counts_layer = counts_layer_arg
    )
  },

  # MacParland et al. 2018, Nat Commun. Human liver, 10x, ~8.4k cells.
  # Source: GEO GSE115469, redistributed on cellxgene.
  # NOTE: If the direct URL becomes stale, download the .h5ad manually
  # from cellxgene.cziscience.com and pass --h5ad_path.
  macparland_human_liver = {
    load_h5ad_sce(
      url = paste0(
        "https://datasets.cellxgene.cziscience.com/",
        "3a2b0148-9c8c-4a3a-8b5f-1c1b1c2b0a01.h5ad"
      ),
      output_dir = output_dir,
      h5ad_path = h5ad_path_arg,
      counts_layer = counts_layer_arg
    )
  },

  stop(
    "Unsupported --dataset: ", dataset, "\n",
    "Supported datasets are: baron_human_pancreas, muraro_human_pancreas, ",
    "segerstolpe_human_pancreas, zeisel_mouse_brain, ",
    "litvinukova_human_heart, macparland_human_liver."
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
  # Common across scRNAseq datasets
  "label",
  "celltype",
  "cell_type",
  "cell type",
  "cell.type",
  "Cell_type",
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
  "secondary_type",
  # Litvinukova heart
  "cell_states",
  "cell_state",
  # HLCA / broader HCA atlases
  "ann_finest_level",
  "ann_level_1", "ann_level_2", "ann_level_3", "ann_level_4",
  # Stewart kidney
  "compartment",
  "broad_celltype",
  "Annotation",
  "annotation_level_1",
  "annotation_level_2",
  # MacParland liver
  "cell_type_manual",
  # HeOrganAtlas / Zilionis / Nowakowski
  "reclustered.broad.cell.type",
  "Major cell type",
  "WGCNAcluster"
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

# Warn if the .h5ad-based datasets look log-normalized (non-integer values).
# Baron / Muraro / Segerstolpe / Zeisel come as raw counts already.
if (dataset %in% c("litvinukova_human_heart",
                   "macparland_human_liver")) {
  sample_vals <- counts@x[seq_len(min(length(counts@x), 10000))]
  if (length(sample_vals) > 0 && any(sample_vals > 0 & sample_vals < 1)) {
    message(
      "WARNING: matrix contains non-integer values in the sampled range. ",
      "This assay may be log-normalized. If so, rerun with ",
      "--counts_layer <name_of_raw_layer> (e.g. --counts_layer counts) ",
      "or supply a raw .h5ad via --h5ad_path."
    )
  }
}

rownames(counts) <- rownames(sce)
colnames(counts) <- colnames(sce)

# -----------------------------
# Write 10x-compatible files
# -----------------------------
matrix_file_plain   <- file.path(dir_10x, "matrix.mtx")
features_file_plain <- file.path(dir_10x, "features.tsv")
barcodes_file_plain <- file.path(dir_10x, "barcodes.tsv")

matrix_file   <- paste0(matrix_file_plain,   ".gz")
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
gene_ids   <- rownames(counts)
gene_names <- gene_ids

rowdata <- as.data.frame(SummarizedExperiment::rowData(sce))

possible_symbol_cols <- c(
  "Symbol", "symbol", "gene_symbol", "gene_name",
  "GeneName", "external_gene_name", "feature_symbol",
  "gene_symbols", "index"
)

matched_symbol_cols <- possible_symbol_cols[possible_symbol_cols %in% colnames(rowdata)]

if (length(matched_symbol_cols) > 0) {
  tmp_gene_names <- as.character(rowdata[[matched_symbol_cols[1]]])
  if (length(tmp_gene_names) == length(gene_ids)) {
    empty <- is.na(tmp_gene_names) | tmp_gene_names == ""
    tmp_gene_names[empty] <- gene_ids[empty]
    gene_names <- tmp_gene_names
  }
}

features_df <- data.frame(
  gene_id      = gene_ids,
  gene_name    = gene_names,
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

matrix_size   <- file.info(matrix_file)$size
features_size <- file.info(features_file)$size
barcodes_size <- file.info(barcodes_file)$size

message("matrix.mtx.gz size: ",   matrix_size,   " bytes")
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
  cell_id                = colnames(counts),
  ground_truth_cell_type = as.character(truth_labels[colnames(counts)]),
  dataset                = dataset,
  external_label_column  = truth_col,
  stringsAsFactors       = FALSE
)

write.csv(metadata_out, metadata_file, row.names = FALSE)

# -----------------------------
# Write summary
# -----------------------------
label_table <- sort(table(metadata_out$ground_truth_cell_type), decreasing = TRUE)

summary_file <- file.path(output_dir, "dataset_summary.txt")

source_str <- if (dataset == "litvinukova_human_heart") {
  "HCA Heart Cell Atlas / heartcellatlas.org (Litvinukova et al. 2020, Nature)"
} else if (dataset == "macparland_human_liver") {
  "GEO GSE115469 / cellxgene (MacParland et al. 2018, Nat Commun)"
} else if (grepl("^stewart_", dataset)) {
  "kidneycellatlas.org / ArrayExpress E-MTAB-8210 (Stewart et al. 2019, Science)"
} else {
  "Bioconductor scRNAseq"
}

summary_lines <- c(
  paste0("Dataset: ", prefix, " / ", dataset),
  paste0("Source: ", source_str),
  paste0("Object class: SingleCellExperiment"),
  paste0("External annotation column: ", truth_col),
  paste0("Number of cells: ", ncol(counts)),
  paste0("Number of genes/features: ", nrow(counts)),
  paste0("Number of external cell types: ", length(label_table)),
  paste0("Missing labels after filtering: ",
         sum(is.na(metadata_out$ground_truth_cell_type) |
               metadata_out$ground_truth_cell_type == "")),
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