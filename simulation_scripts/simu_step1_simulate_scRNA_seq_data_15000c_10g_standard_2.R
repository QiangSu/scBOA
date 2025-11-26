#!/usr/bin/env Rscript
print("Starting Simulation Pipeline: Counts with Splatter and FASTQ with Polyester")

# --- 0: Set Seed for Reproducibility ---
set.seed(123)

# --- 1: Load Libraries ---
print("Loading required libraries...")
# Ensure libraries are installed (uncomment if needed)
# if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
# if (!requireNamespace("splatter", quietly = TRUE)) BiocManager::install("splatter")
# if (!requireNamespace("polyester", quietly = TRUE)) BiocManager::install("polyester")
# if (!requireNamespace("Biostrings", quietly = TRUE)) BiocManager::install("Biostrings")

library(splatter)
library(polyester)
library(Biostrings)
library(SummarizedExperiment) # Ensure SummarizedExperiment is loaded for counts() and colData()
library(Matrix)             # Explicitly load Matrix package for sparse matrix handling

# --- 2: Define Paths ---
base_dir <- "/home/data/qs/scRNA_simulation_data/simulated_data_15000c_10g_standard_2"
dir.create(base_dir, showWarnings = FALSE)
reference_fasta <- file.path("/home/data/qs/scRNA_simulation_data/Mus_musculus.GRCm38.cdna.all.fa")
ground_truth_matrix_file <- file.path(base_dir, "ground_truth_matrix.csv")
ground_truth_celltypes_file <- file.path(base_dir, "ground_truth_celltypes.txt")
output_fastq_dir <- file.path(base_dir, "simulated_fastq_15000c_10g_standard")
dir.create(output_fastq_dir, showWarnings = FALSE)

# --- 3: Load Reference Transcript IDs ---
print("Loading reference transcript IDs...")
if (file.exists(reference_fasta)) {
    # Read FASTA file headers to extract transcript IDs
    fasta_data <- readDNAStringSet(reference_fasta) # Renamed to avoid conflict
    # Extract headers and take only the first part before any space (e.g., "ENSMUST00000123456.1")
    transcript_ids <- sapply(strsplit(names(fasta_data), " "), `[`, 1)
    # Clean up IDs by removing version numbers after "."
    transcript_ids <- sub("\\..*", "", transcript_ids) # e.g., "ENSMUST00000123456.1" -> "ENSMUST00000123456"
    print(paste("Loaded", length(unique(transcript_ids)), "unique transcript IDs from FASTA file."))
    transcript_ids <- unique(transcript_ids) # Ensure uniqueness
} else {
    # Use a shorter list of dummy IDs if reference is not found - but it's better to stop
    # warning("Reference FASTA file not found. Using dummy transcript IDs for testing.")
    # transcript_ids <- paste0("ENST_dummy_", 1:30000) # Create 30,000 dummy IDs
    # print(paste("Generated", length(transcript_ids), "dummy transcript IDs."))
    stop(paste("Reference FASTA file not found:", reference_fasta)) # Stop if reference is missing
}

# --- 4: Simulate Counts with Splatter (Step 1) ---
print("Simulating single-cell RNA-seq counts with Splatter...")

# Define simulation parameters
n_cells <- 15000        # Number of cells
n_genes <- min(50000, length(transcript_ids))  # Number of genes (limit to available unique transcript IDs)
n_groups <- 10         # Number of cell types/clusters

# Ensure we only select genes present in the reference
available_genes <- transcript_ids
if (n_genes > length(available_genes)) {
    warning(paste("Target n_genes", n_genes, "is greater than available unique transcript IDs", length(available_genes), ". Adjusting n_genes."))
    n_genes <- length(available_genes)
}
selected_gene_ids <- sample(available_genes, n_genes, replace = FALSE) # Select genes BEFORE simulation

# Set Splatter parameters for realistic simulation with specified groups
params <- newSplatParams()
params <- setParam(params, "nGenes", n_genes)
params <- setParam(params, "batchCells", n_cells)
params <- setParam(params, "group.prob", rep(1/n_groups, n_groups)) # Define groups with equal probability
# params <- setParam(params, "de.prob", 0.5)    # 30% of genes are differentially expressed
# params <- setParam(params, "de.facLoc", 3)    # DE factor location (controls magnitude of DE)
# params <- setParam(params, "de.facScale", 0.2) # DE factor scale
# params <- setParam(params, "bcv.common", 0.05) # Reduce within-cluster variability

params <- setParam(params, "de.prob", 1)    # 30% of genes are differentially expressed
params <- setParam(params, "de.facLoc", 3)    # DE factor location (controls magnitude of DE)
params <- setParam(params, "de.facScale", 0.6) # DE factor scale
params <- setParam(params, "bcv.common", 0.2) # Reduce within-cluster variability
params <- setParam(params, "bcv.df", 30)     # Decrease DF from default 60

params <- setParam(params, "dropout.type", "experiment") # Dropout simulation
params <- setParam(params, "dropout.mid", 0)  # Midpoint for dropout logistic
params <- setParam(params, "dropout.shape", -1) # Shape for dropout logistic
params <- setParam(params, "seed", 123)       # Add seed to splatter params too

# Simulate the data
sim_data <- splatSimulate(params, method = "groups", verbose = TRUE)

# Extract count matrix
counts <- counts(sim_data)

# --- Important: Assign the *selected* gene IDs ---
# Splatter might generate its own GeneX names, override them
if(nrow(counts) == length(selected_gene_ids)) {
    rownames(counts) <- selected_gene_ids
    print("Assigned selected transcript IDs as gene names to the count matrix.")
} else {
    warning("Row count mismatch after Splatter simulation. Cannot assign selected gene IDs.")
    # Consider stopping if IDs cannot be assigned correctly
    # stop("Cannot proceed without correct gene IDs assignment.")
}

# Set column names (Cell IDs)
# These names will be used in the ground truth files
colnames(counts) <- paste0("Cell_", 1:n_cells)

# Extract cell type assignments
sim_cell_types <- colData(sim_data)$Group
names(sim_cell_types) <- colnames(counts) # Assign Cell IDs to the types vector

# Save outputs
print(paste("Saving ground truth count matrix to", ground_truth_matrix_file))
# Convert sparse matrix to dense matrix before saving <<< FIX APPLIED HERE
write.csv(as.matrix(counts), file = ground_truth_matrix_file, quote = FALSE)

print(paste("Saving ground truth cell types to", ground_truth_celltypes_file))
# Save cell types with Cell IDs and a header
write.table(data.frame(CellID=names(sim_cell_types), CellType=sim_cell_types),
            file = ground_truth_celltypes_file,
            sep="\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

print("Step 4 (using Splatter) complete.")


# --- 5: Simulate FASTQ Files with Polyester (Step 2) ---
print("Starting Step 5: Simulate FASTQ Files with Polyester")

# Reload reference transcriptome to ensure consistency
print("Loading reference transcriptome for FASTQ simulation...")
ref_transcriptome <- readDNAStringSet(reference_fasta)
# Extract transcript IDs from headers, taking only the first part before any space
ref_transcript_ids <- sapply(strsplit(names(ref_transcriptome), " "), `[`, 1)
# Clean up IDs by removing version numbers after "."
ref_transcript_ids <- sub("\\..*", "", ref_transcript_ids) # Match Step 4's cleaning
names(ref_transcriptome) <- ref_transcript_ids # Update names to cleaned IDs

# Make sure names are unique in the reference (can happen if multiple isoforms map to same gene ID after cleaning)
ref_transcriptome <- ref_transcriptome[!duplicated(names(ref_transcriptome))]
ref_transcript_ids <- names(ref_transcriptome)
print(paste("Using", length(ref_transcript_ids), "unique reference sequences after ID cleaning."))

print("Loading simulated count matrix for FASTQ simulation...")
# Reload the matrix, ensure row names are kept
counts_for_polyester <- read.csv(ground_truth_matrix_file, row.names = 1, check.names = FALSE)
# Polyester expects column names that match the number of samples (1:n_cells)
# It might use these to generate filenames like sample_01, sample_10 etc.
colnames(counts_for_polyester) <- as.character(1:n_cells) # Use simple numbers for Polyester

# Match transcripts between counts and reference
common_transcripts <- intersect(rownames(counts_for_polyester), ref_transcript_ids)
if (length(common_transcripts) == 0) {
    stop("No matching transcripts between count matrix and reference FASTA. Check gene/transcript IDs.")
}
print(paste("Found", length(common_transcripts), "common transcripts between counts and reference for simulation."))

# Ensure matrix and reference only contain common transcripts *in the same order*
counts_for_polyester <- counts_for_polyester[common_transcripts, ]
ref_transcriptome_filtered <- ref_transcriptome[common_transcripts] # Use a new name

# Save filtered transcriptome as a temporary FASTA file to pass to polyester
temp_fasta_file <- file.path(base_dir, "temp_filtered_transcriptome.fa")
print(paste("Saving filtered transcriptome (ordered) to temporary file:", temp_fasta_file))
writeXStringSet(ref_transcriptome_filtered, filepath = temp_fasta_file, format = "fasta")

# Simulate paired-end reads using the temporary FASTA file path
# Explicitly set parameters to ensure FASTQ output with quality scores
print("Simulating FASTQ files with Polyester...")
simulation_successful <- FALSE
tryCatch({
    simulate_experiment_countmat(fasta = temp_fasta_file,        # Pass file path
                                 readmat = as.matrix(counts_for_polyester), # Ensure matrix format
                                 outdir = output_fastq_dir,
                                 paired = TRUE,
                                 readlen = 150,
                                 error_rate = 0.005,
                                 distr = "normal", # Default distribution for fragment lengths
                                 meanmodel = TRUE, # Use a model for fragment length mean
                                 seed = 123)       # Different seed for this step if desired
    print(paste("Step 5: FASTQ simulation complete. Raw files potentially saved to:", output_fastq_dir))
    simulation_successful <- TRUE
}, error = function(e) {
    print("Error during FASTQ simulation:")
    print(e)
    stop("FASTQ simulation failed. Check error message above for details.")
})

# Clean up temporary FASTA file
if (file.exists(temp_fasta_file)) {
    unlink(temp_fasta_file)
    print("Temporary FASTA file removed.")
}

# --- 6: Rename Polyester Output Files ---
# Goal: Convert sample_0X_Y.fasta to sample_X_Y.fastq
#       Convert sample_XY_Z.fasta to sample_XY_Z.fastq
if (simulation_successful) {
    print("Step 6: Renaming Polyester output files (.fasta -> .fastq, removing leading zeros)...")

    # List all files matching the Polyester output pattern with .fasta extension
    files_to_process <- list.files(output_fastq_dir,
                                   pattern = "^sample_[0-9]+_(1|2)\\.fasta$",
                                   full.names = TRUE)

    if (length(files_to_process) == 0) {
        print("No .fasta files matching 'sample_...fasta' pattern found to rename.")
    } else {
        renamed_count <- 0
        processed_count <- 0
        failed_renames <- c()

        print(paste("Found", length(files_to_process), "files to process."))

        for (old_path in files_to_process) {
            processed_count <- processed_count + 1
            old_basename <- basename(old_path)

            # Use regex to extract parts and rebuild the name
            # Match 'sample_', then capture the number part, then '_', capture read pair, then '.fasta'
            match_result <- regexpr("^sample_([0-9]+)_([12])\\.fasta$", old_basename, perl = TRUE)

            if (attr(match_result, "capture.start")[1] > 0) { # Check if regex matched
                # Extract captured groups
                sample_num_str <- substr(old_basename,
                                         attr(match_result, "capture.start")[1],
                                         attr(match_result, "capture.start")[1] + attr(match_result, "capture.length")[1] - 1)
                read_pair <- substr(old_basename,
                                    attr(match_result, "capture.start")[2],
                                    attr(match_result, "capture.start")[2] + attr(match_result, "capture.length")[2] - 1)

                # Remove leading zero if present (convert to number and back to string)
                sample_num_corrected <- as.character(as.integer(sample_num_str))

                # Construct the new filename with .fasta extension
                target_basename <- paste0("sample_", sample_num_corrected, "_", read_pair, ".fasta")

                if (target_basename != old_basename) {
                    target_path <- file.path(dirname(old_path), target_basename)
                    print(paste("Renaming:", old_basename, "->", target_basename))
                    rename_success <- try(file.rename(from = old_path, to = target_path), silent = TRUE)

                    if (inherits(rename_success, "try-error") || !rename_success) {
                        warning(paste("Failed to rename", old_basename, "to", target_basename))
                        failed_renames <- c(failed_renames, old_basename)
                    } else {
                        renamed_count <- renamed_count + 1
                    }
                } else {
                     # This case shouldn't happen if the extension is always changing from .fasta to .fastq
                     print(paste("Skipping", old_basename, "- target name is the same (unexpected)."))
                }
            } else {
                warning(paste("File", old_basename, "did not match expected pattern for renaming."))
            }
        }
        print(paste("Finished processing", processed_count, "files found with .fasta extension."))
        print(paste("Successfully renamed", renamed_count, "files to .fastq format (removing leading zeros where applicable)."))
        if (length(failed_renames) > 0) {
            warning("Failed attempts for the following original files: ", paste(failed_renames, collapse=", "))
        }
    }
} else {
    print("Skipping renaming step due to simulation failure.")
}

# --- 7: Summary ---
print("Simulation pipeline complete.")
print("Outputs generated:")
print(paste("- Ground truth count matrix:", ground_truth_matrix_file))
print(paste("- Ground truth cell types:", ground_truth_celltypes_file))
print(paste("- Simulated FASTQ files (check names in):", output_fastq_dir))

