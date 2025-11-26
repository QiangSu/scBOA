#!/usr/bin/env Rscript

# ==============================================================================
# --- Quad-Objective Standalone R script for single-cell RNA-seq analysis ---
# ==============================================================================
#
# DESCRIPTION:
# This script performs Bayesian optimization to find optimal scRNA-seq analysis
# parameters (HVGs, PCs, n_neighbors, resolution) using Seurat's cross-dataset
# anchoring for cell type annotation. It requires a pre-annotated Seurat object
# to serve as a reference for label transfer.
#
# It supports two optimization models selectable via the '--model_type' argument:
# 1. 'biological': Optimizes for a balanced score combining weighted_cas,
#    simple_cas, and mcs using a geometric mean.
# 2. 'structural': Optimizes for a balanced score combining the three biological
#    metrics PLUS the silhouette score for cluster quality.
#
# It supports both HUMAN and MOUSE data, selectable via the '--species' argument.
# This choice determines the gene annotation database and mitochondrial gene prefix.
#
# For each optimization target, this script automatically runs THREE parallel
# Bayesian optimization strategies to compare their search behavior:
# 1. 'Exploit' (Exploitation-focused using 'POI')
# 2. 'Explore' (Exploration-focused using 'EI' with high jitter)
# 3. 'BO-EI' (Balanced Bayesian Optimization using 'EI')
#
# It produces a comprehensive set of outputs for each target:
# - A t-SNE plot of the search space showing the paths of the optimizers.
# - A convergence plot comparing the performance of each optimizer over time.
# - A consolidated CSV file with detailed results for every trial.
# - A final text report and an annotated Seurat Object (.rds) with the best parameters.
#
# HOW TO RUN:
# 1. Ensure you have an annotated reference Seurat object (.rds file).
#
# 2. Run the R script from your terminal (example for human data):
#    Rscript this_script_name.R --species human --data_dir /path/to/10x/ \
#    --output_dir ./output --reference_path /path/to/reference.rds \
#    --reference_labels_col celltype_column
#
# 3. Example for mouse data:
#    Rscript this_script_name.R --species mouse --data_dir /path/to/10x/ ...
#
# 4. (RECOMMENDED FOR SPEED) Use subsampling for optimization:
#    Rscript ... --subsample_n_cells 5000

# --- Load Libraries ---
suppressPackageStartupMessages({
    library(Seurat)
    library(dplyr)
    library(ggplot2)
    library(argparse)
    library(rBayesianOptimization)
    library(Rtsne)
    library(cluster)
    library(AnnotationDbi)
    # Note: org.Hs.eg.db and org.Mm.eg.db are loaded dynamically based on --species flag
})

# ==============================================================================
# --- *** CONFIGURATION SECTION *** ---
# ==============================================================================
# --- [MODIFICATION] MITO_PREFIX is now set dynamically in main() ---
MIN_GENES_PER_CELL <- 200
MAX_GENES_PER_CELL <- 7000
MAX_PCT_COUNTS_MT <- 10
MIN_CELLS_PER_GENE <- 3

# Search space for Bayesian Optimization
search_space_bounds <- list(
    n_hvg = c(500L, 20000L),
    n_pcs = c(10L, 100L),
    n_neighbors = c(10L, 50L),
    resolution = c(0.2, 2.0)
)
N_PCS_FOR_PCA <- 105
N_MCS_TOP_GENES <- 5

# --- Global variables ---
seurat_base <- NULL
seurat_full_data <- NULL
seurat_ref <- NULL
REF_LABELS_COL <- NULL
RANDOM_SEED <- NULL
ARGS <- NULL
CURRENT_OPTIMIZATION_TARGET <- NULL
CURRENT_STRATEGY_NAME <- ""
TRIAL_METADATA <- list()
OPTIMIZATION_CACHE <- NULL 

# --- [MODIFICATION] Species-specific globals ---
SPECIES_DB <- NULL
MITO_PREFIX <- NULL

# ==============================================================================
# --- *** HELPER TO DYNAMICALLY LOAD PACKAGES *** ---
# ==============================================================================

# --- [NEW FUNCTION] ---
# Checks for a package, installs it from Bioconductor if missing, then loads it.
check_and_load_bioc_package <- function(pkg_name) {
    if (!require(pkg_name, character.only = TRUE)) {
        cat(sprintf("   [INFO] Package '%s' not found. Attempting to install from Bioconductor...\n", pkg_name))
        if (!requireNamespace("BiocManager", quietly = TRUE)) {
            install.packages("BiocManager")
        }
        BiocManager::install(pkg_name, update = FALSE, ask = FALSE)
        
        # Load the package after installation
        if (!require(pkg_name, character.only = TRUE)) {
            stop(sprintf("Failed to install or load package: %s. Please install it manually.", pkg_name))
        }
    }
}


# ==============================================================================
# --- *** OBJECTIVE FUNCTION (UNCHANGED) *** ---
# ==============================================================================
objective_function <- function(n_hvg, n_pcs, n_neighbors, resolution) {
    # Ensure integer parameters are integers
    n_hvg <- as.integer(round(n_hvg))
    n_pcs <- as.integer(round(n_pcs))
    n_neighbors <- as.integer(round(n_neighbors))

    start_time <- Sys.time()

    # --- Caching & Parameter Discretization ---
    cache_key <- sprintf("hvg%d_pcs%d_nei%d_res%.4f", n_hvg, n_pcs, n_neighbors, resolution)

    if (exists(cache_key, envir = OPTIMIZATION_CACHE)) {
        cached_result <- OPTIMIZATION_CACHE[[cache_key]]
        
        # We still print the summary for cached hits for full transparency
        cached_params <- list(n_hvg=n_hvg, n_pcs=n_pcs, n_neighbors=n_neighbors, resolution=resolution)
        cached_metrics <- cached_result$Metrics
        cached_score <- cached_result$Score
        
        cat("\n")
        cat("     +-------------------- CACHED TRIAL ---------------------+\n")
        cat(sprintf("     | Strategy:   %-15s | (Result from cache)\n", CURRENT_STRATEGY_NAME))
        cat("     +---------------------------------------------------------+\n")
        cat(sprintf("     | Parameters:\n"))
        cat(sprintf("     |   n_hvg:       %d\n", cached_params$n_hvg))
        cat(sprintf("     |   n_pcs:       %d\n", cached_params$n_pcs))
        cat(sprintf("     |   n_neighbors: %d\n", cached_params$n_neighbors))
        cat(sprintf("     |   resolution:  %.4f\n", cached_params$resolution))
        cat("     |---------------------------------------------------------+\n")
        cat(sprintf("     | Individual Metrics:\n"))
        cat(sprintf("     |   Weighted Mean CAS:   %.2f %%\n", cached_metrics$weighted_mean_cas))
        cat(sprintf("     |   Simple Mean CAS:     %.2f %%\n", cached_metrics$simple_mean_cas))
        cat(sprintf("     |   Mean MCS:            %.2f %%\n", cached_metrics$mean_mcs))
        cat(sprintf("     |   Silhouette Score:    %.4f\n", cached_metrics$silhouette_score))
        cat("     +---------------------------------------------------------+\n")
        cat(sprintf("     | ==> FINAL SCORE (Target: %s): %.4f\n", CURRENT_OPTIMIZATION_TARGET, cached_score))
        cat("     +---------------------------------------------------------+\n\n")

        if (!is.null(cached_result$Metrics)) {
             TRIAL_METADATA[[length(TRIAL_METADATA) + 1]] <<- cached_result$Metrics
        }
        
        jitter <- rnorm(1, mean = 0, sd = 1e-2)
        return(list(Score = cached_result$Score + jitter, Pred = 0))
    }

    # This function now naturally uses the smaller 'seurat_base' if subsampling is active
    seurat_proc <- seurat_base

    is_two_step_hvg <- !is.null(ARGS$hvg_min_mean) && !is.null(ARGS$hvg_max_mean) && !is.null(ARGS$hvg_min_disp)

    if (is_two_step_hvg) {
        seurat_proc <- FindVariableFeatures(seurat_proc, method = "vst", nfeatures = nrow(seurat_proc), verbose = FALSE)
        hvg_info <- HVFInfo(seurat_proc, method = "vst", assay = "RNA")
        
        hvg_info_filtered <- subset(hvg_info,
                                    mean > ARGS$hvg_min_mean &
                                    mean < ARGS$hvg_max_mean &
                                    variance.standardized > ARGS$hvg_min_disp)
        
        hvg_info_sorted <- hvg_info_filtered[order(-hvg_info_filtered$variance.standardized), ]
        
        n_hvg_safe <- min(n_hvg, nrow(hvg_info_sorted))
        if (n_hvg_safe < n_hvg) {
            cat(sprintf("     [WARNING] Requested %d HVGs, but only %d passed the filter. Using %d.\n", n_hvg, n_hvg_safe, n_hvg_safe))
        }
        
        top_genes <- rownames(hvg_info_sorted)[1:n_hvg_safe]
        VariableFeatures(seurat_proc) <- top_genes
        
    } else {
        seurat_proc <- FindVariableFeatures(seurat_proc, method = "vst", nfeatures = n_hvg, verbose = FALSE)
    }

    if (length(VariableFeatures(seurat_proc)) == 0) {
        cat("     [CRITICAL WARNING] HVG selection resulted in 0 genes. Returning score 0.\n")
        return(list(Score = 0, Pred = 0))
    }

    seurat_proc <- ScaleData(seurat_proc, features = rownames(seurat_proc), verbose = FALSE)
    seurat_proc <- RunPCA(seurat_proc, npcs = N_PCS_FOR_PCA, features = VariableFeatures(object = seurat_proc), verbose = FALSE, seed.use = RANDOM_SEED)
    seurat_proc <- FindNeighbors(seurat_proc, dims = 1:n_pcs, k.param = n_neighbors, verbose = FALSE)
    seurat_proc <- FindClusters(seurat_proc, resolution = resolution, algorithm = 4, # 4 = Leiden
                                random.seed = RANDOM_SEED, verbose = FALSE)
    seurat_proc$leiden <- seurat_proc$seurat_clusters

    rescaled_silhouette <- 0.0
    tryCatch({
        n_clusters <- nlevels(seurat_proc$leiden)
        if (n_clusters > 1 && n_clusters < ncol(seurat_proc)) {
            pca_coords <- Embeddings(seurat_proc, reduction = "pca")[, 1:n_pcs]
            dist_matrix <- dist(pca_coords)
            sil_scores <- silhouette(as.integer(seurat_proc$leiden), dist_matrix)
            silhouette_avg <- mean(sil_scores[, "sil_width"])
            rescaled_silhouette <- (silhouette_avg + 1) / 2
        }
    }, error = function(e) {
        cat(sprintf("     [WARNING] Silhouette score calculation failed. Error: %s. Score set to 0.\n", e$message))
        rescaled_silhouette <<- 0.0
    })

    weighted_mean_cas <- 0.0
    simple_mean_cas <- 0.0
    tryCatch({
        transfer_anchors <- FindTransferAnchors(
            reference = seurat_ref,
            query = seurat_proc,
            dims = 1:n_pcs,
            reduction = "pcaproject",
            reference.assay = ARGS$reference_assay,
            query.assay = "RNA",
            verbose = FALSE
        )

        predictions <- TransferData(
            anchorset = transfer_anchors,
            refdata = seurat_ref[[REF_LABELS_COL, drop = TRUE]],
            dims = 1:n_pcs,
            k.weight = n_neighbors,
            weight.reduction = "pcaproject",
            verbose = FALSE
        )

        seurat_proc$ctpt_individual_prediction <- predictions$predicted.id

        metadata_df <- seurat_proc@meta.data %>%
            group_by(leiden) %>%
            mutate(ctpt_consensus_prediction = names(which.max(table(ctpt_individual_prediction)))) %>%
            ungroup()

        total_cells <- nrow(metadata_df)
        total_matching <- sum(metadata_df$ctpt_individual_prediction == metadata_df$ctpt_consensus_prediction, na.rm = TRUE)
        weighted_mean_cas <- (total_matching / total_cells) * 100

        cas_per_cluster <- metadata_df %>%
            group_by(leiden) %>%
            summarise(cas = mean(ctpt_individual_prediction == dplyr::first(ctpt_consensus_prediction)) * 100)
        simple_mean_cas <- mean(cas_per_cluster$cas, na.rm = TRUE)

        seurat_proc$ctpt_consensus_prediction <- metadata_df$ctpt_consensus_prediction
    }, error = function(e) {
        cat(sprintf("     [WARNING] Seurat anchoring or CAS calculation failed. Error: %s. Scores set to 0.\n", e$message))
        weighted_mean_cas <<- 0.0
        simple_mean_cas <<- 0.0
    })

    mean_mcs <- 0.0
    tryCatch({
        if ("ctpt_consensus_prediction" %in% colnames(seurat_proc@meta.data)) {
            Idents(seurat_proc) <- "ctpt_consensus_prediction"
            label_counts <- table(seurat_proc$ctpt_consensus_prediction)
            valid_labels <- names(label_counts[label_counts > 1])

            if (length(valid_labels) >= 2) {
                markers <- FindAllMarkers(seurat_proc, assay = "RNA", only.pos = TRUE, min.pct = 0.25,
                                          logfc.threshold = 0, verbose = FALSE, layer = "data")

                if (nrow(markers) > 0) {
                    top_genes_per_group <- markers %>% group_by(cluster) %>% top_n(n = N_MCS_TOP_GENES, wt = avg_log2FC)
                    unique_top_genes <- unique(top_genes_per_group$gene)
                    if (length(unique_top_genes) > 0) {
                        expr_data <- FetchData(seurat_proc, vars = c("ctpt_consensus_prediction", unique_top_genes))
                        fraction_df <- expr_data %>% group_by(ctpt_consensus_prediction) %>% summarise(across(all_of(unique_top_genes), ~ mean(.x > 0)))
                        mcs_scores <- list()
                        for (cell_type in unique(top_genes_per_group$cluster)) {
                            markers_for_type <- top_genes_per_group %>% filter(cluster == cell_type) %>% pull(gene)
                            prevalence_values <- fraction_df %>% filter(ctpt_consensus_prediction == cell_type) %>% dplyr::select(all_of(markers_for_type))
                            mcs_scores[[cell_type]] <- mean(as.numeric(prevalence_values))
                        }
                        mean_mcs <- mean(unlist(mcs_scores), na.rm = TRUE) * 100
                    }
                }
            }
        }
    }, error = function(e) {
        cat(sprintf("     [WARNING] MCS calculation failed. Error: %s. MCS set to 0.\n", e$message))
        mean_mcs <<- 0.0
    })

    if (is.nan(weighted_mean_cas)) weighted_mean_cas <- 0.0
    if (is.nan(simple_mean_cas)) simple_mean_cas <- 0.0
    if (is.nan(mean_mcs)) mean_mcs <- 0.0
    if (is.nan(rescaled_silhouette)) rescaled_silhouette <- 0.0

    trial_data <- list(
        n_individual_labels = if ("ctpt_individual_prediction" %in% colnames(seurat_proc@meta.data)) n_distinct(seurat_proc$ctpt_individual_prediction) else 0,
        n_consensus_labels = if ("ctpt_consensus_prediction" %in% colnames(seurat_proc@meta.data)) n_distinct(seurat_proc$ctpt_consensus_prediction) else 0,
        weighted_mean_cas = weighted_mean_cas, simple_mean_cas = simple_mean_cas, mean_mcs = mean_mcs, silhouette_score = rescaled_silhouette
    )
    TRIAL_METADATA[[length(TRIAL_METADATA) + 1]] <<- trial_data

    score <- 0.0
    if (CURRENT_OPTIMIZATION_TARGET == 'weighted_cas') {
        score <- weighted_mean_cas
    } else if (CURRENT_OPTIMIZATION_TARGET == 'simple_cas') {
        score <- simple_mean_cas
    } else if (CURRENT_OPTIMIZATION_TARGET == 'mcs') {
        score <- mean_mcs
    } else if (CURRENT_OPTIMIZATION_TARGET == 'balanced') {
        epsilon <- 1e-6
        if (ARGS$model_type == 'structural') {
            score <- (((weighted_mean_cas / 100 + epsilon) * (simple_mean_cas / 100 + epsilon) * (mean_mcs / 100 + epsilon) * (rescaled_silhouette + epsilon)) ^ (1/4.0)) * 100
        } else {
            score <- (((weighted_mean_cas / 100 + epsilon) * (simple_mean_cas / 100 + epsilon) * (mean_mcs / 100 + epsilon)) ^ (1/3.0)) * 100
        }
    } else {
        stop(paste("Invalid optimization target:", CURRENT_OPTIMIZATION_TARGET))
    }
    
    end_time <- Sys.time()
    time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))

    # --- Comprehensive summary box for this trial ---
    cat("\n")
    cat("     +--------------------- TRIAL SUMMARY ---------------------+\n")
    cat(sprintf("     | Strategy:   %-15s | Time: %.1f sec\n", CURRENT_STRATEGY_NAME, time_taken))
    cat("     +---------------------------------------------------------+\n")
    cat(sprintf("     | Parameters:\n"))
    cat(sprintf("     |   n_hvg:       %d\n", n_hvg))
    cat(sprintf("     |   n_pcs:       %d\n", n_pcs))
    cat(sprintf("     |   n_neighbors: %d\n", n_neighbors))
    cat(sprintf("     |   resolution:  %.4f\n", resolution))
    cat("     |---------------------------------------------------------+\n")
    cat(sprintf("     | Individual Metrics:\n"))
    cat(sprintf("     |   Weighted Mean CAS:   %.2f %%\n", weighted_mean_cas))
    cat(sprintf("     |   Simple Mean CAS:     %.2f %%\n", simple_mean_cas))
    cat(sprintf("     |   Mean MCS:            %.2f %%\n", mean_mcs))
    cat(sprintf("     |   Silhouette Score:    %.4f\n", rescaled_silhouette))
    cat("     +---------------------------------------------------------+\n")
    cat(sprintf("     | ==> FINAL SCORE (Target: %s): %.4f\n", CURRENT_OPTIMIZATION_TARGET, score))
    cat("     +---------------------------------------------------------+\n\n")

    OPTIMIZATION_CACHE[[cache_key]] <- list(Score = score, Metrics = trial_data)

    jitter <- rnorm(1, mean = 0, sd = 1e-2)
    final_score_for_optimizer <- score + jitter
    
    if (!is.finite(final_score_for_optimizer)) {
        cat(sprintf("     [CRITICAL WARNING] Non-finite score detected. Returning 0.\n"))
        return(list(Score = 0, Pred = 0))
    }
    
    return(list(Score = final_score_for_optimizer, Pred = 0))
}

# ==============================================================================
# --- *** HELPER FUNCTIONS FOR REPORTING & SAVING (UNCHANGED) *** ---
# ==============================================================================

evaluate_final_metrics <- function(params) {
    cat("\n--- Re-running analysis with overall best parameters for final report ---\n")
    
    # --- This function now uses the FULL dataset ---
    seurat_final <- .GlobalEnv$seurat_full_data
    
    cat(sprintf("[INFO] Using the FULL dataset of %d cells for the final run.\n", ncol(seurat_final)))

    params$n_hvg <- as.integer(params$n_hvg)
    params$n_pcs <- as.integer(params$n_pcs)
    params$n_neighbors <- as.integer(params$n_neighbors)
    
    is_two_step_hvg <- !is.null(ARGS$hvg_min_mean) && !is.null(ARGS$hvg_max_mean) && !is.null(ARGS$hvg_min_disp)

    if (is_two_step_hvg) {
        cat("[INFO] Using two-step sequential HVG selection for final object.\n")
        seurat_final <- FindVariableFeatures(seurat_final, method = "vst", nfeatures = nrow(seurat_final), verbose = FALSE)
        hvg_info <- HVFInfo(seurat_final, method = "vst", assay = "RNA")
        
        hvg_info_filtered <- subset(hvg_info,
                                    mean > ARGS$hvg_min_mean &
                                    mean < ARGS$hvg_max_mean &
                                    variance.standardized > ARGS$hvg_min_disp)
        
        hvg_info_sorted <- hvg_info_filtered[order(-hvg_info_filtered$variance.standardized), ]
        
        n_hvg_safe <- min(params$n_hvg, nrow(hvg_info_sorted))
        top_genes <- rownames(hvg_info_sorted)[1:n_hvg_safe]
        VariableFeatures(seurat_final) <- top_genes
        
    } else {
        cat(sprintf("[INFO] Using standard rank-based HVG selection with nfeatures = %d for final object.\n", params$n_hvg))
        seurat_final <- FindVariableFeatures(seurat_final, method = "vst", nfeatures = params$n_hvg, verbose = FALSE)
    }

    all.genes <- rownames(seurat_final)
    seurat_final <- ScaleData(seurat_final, features = all.genes, verbose = FALSE)
    seurat_final <- RunPCA(seurat_final, npcs = N_PCS_FOR_PCA, features = VariableFeatures(object = seurat_final), verbose = FALSE, seed.use = RANDOM_SEED)
    seurat_final <- FindNeighbors(seurat_final, dims = 1:params$n_pcs, k.param = params$n_neighbors, verbose = FALSE)
    seurat_final <- FindClusters(seurat_final, resolution = params$resolution, algorithm = 4, random.seed = RANDOM_SEED, verbose = FALSE)
    seurat_final$leiden <- seurat_final$seurat_clusters
    seurat_final <- RunUMAP(seurat_final, dims = 1:params$n_pcs, seed.use = RANDOM_SEED, verbose = FALSE)

    rescaled_silhouette <- 0.0
    tryCatch({
      if (nlevels(seurat_final$leiden) > 1 && nlevels(seurat_final$leiden) < ncol(seurat_final)) {
        sil <- silhouette(as.integer(seurat_final$leiden), dist(Embeddings(seurat_final, "pca")[, 1:params$n_pcs]))
        rescaled_silhouette <- (mean(sil[, 3]) + 1) / 2
      }
    }, error = function(e){ rescaled_silhouette <<- 0.0 })

    weighted_cas <- 0.0
    simple_cas <- 0.0
    tryCatch({
        transfer_anchors_final <- FindTransferAnchors(
            reference = seurat_ref,
            query = seurat_final,
            dims = 1:params$n_pcs,
            reduction = "pcaproject",
            reference.assay = ARGS$reference_assay,
            query.assay = "RNA",
            verbose = FALSE
        )
        
        predictions_final <- TransferData(
            anchorset = transfer_anchors_final,
            refdata = seurat_ref[[REF_LABELS_COL, drop = TRUE]],
            dims = 1:params$n_pcs,
            k.weight = params$n_neighbors,
            weight.reduction = "pcaproject",
            verbose = FALSE
        )
        
        seurat_final$ctpt_individual_prediction <- predictions_final$predicted.id
        
        metadata_df <- seurat_final@meta.data %>%
            group_by(leiden) %>%
            mutate(ctpt_consensus_prediction = names(which.max(table(ctpt_individual_prediction)))) %>%
            ungroup()
        seurat_final$ctpt_consensus_prediction <- metadata_df$ctpt_consensus_prediction

        weighted_cas <- mean(metadata_df$ctpt_individual_prediction == metadata_df$ctpt_consensus_prediction, na.rm=TRUE) * 100
        simple_cas <- metadata_df %>% group_by(leiden) %>% summarise(cas = mean(ctpt_individual_prediction == dplyr::first(ctpt_consensus_prediction))) %>% pull(cas) %>% mean(na.rm=TRUE) * 100
    }, error = function(e){
        cat(sprintf("     [WARNING] Final CAS calculation via anchoring failed. Error: %s. Scores set to 0.\n", e$message))
        weighted_cas <- 0.0
        simple_cas <- 0.0
    })

    mean_mcs <- 0.0
    tryCatch({
      Idents(seurat_final) <- "ctpt_consensus_prediction"
      label_counts <- table(seurat_final$ctpt_consensus_prediction)
      valid_labels <- names(label_counts[label_counts > 1])
      if (length(valid_labels) >= 2) {
        markers <- FindAllMarkers(seurat_final,
                                  assay = "RNA",
                                  only.pos = TRUE,
                                  min.pct = 0.25,
                                  logfc.threshold = 0, 
                                  verbose = FALSE,
                                  layer = "data") 
        if (nrow(markers) > 0) {
            top_genes <- markers %>% group_by(cluster) %>% top_n(n = N_MCS_TOP_GENES, wt = avg_log2FC)
            unique_genes <- unique(top_genes$gene)
            expr_data <- FetchData(seurat_final, vars = c("ctpt_consensus_prediction", unique_genes))
            frac_df <- expr_data %>% group_by(ctpt_consensus_prediction) %>% summarise(across(all_of(unique_genes), ~ mean(.x > 0)))
            mcs_scores <- sapply(unique(top_genes$cluster), function(ct) {
              m <- top_genes %>% filter(cluster == ct) %>% pull(gene)
              mean(as.numeric(frac_df[frac_df$ctpt_consensus_prediction == ct, m]))
            })
            mean_mcs <- mean(mcs_scores, na.rm=TRUE) * 100
        }
      }
    }, error = function(e){ mean_mcs <<- 0.0 })
    
    if(is.nan(weighted_cas)) weighted_cas <- 0.0
    if(is.nan(simple_cas)) simple_cas <- 0.0
    if(is.nan(mean_mcs)) mean_mcs <- 0.0
    if(is.nan(rescaled_silhouette)) rescaled_silhouette <- 0.0

    epsilon <- 1e-6
    if (ARGS$model_type == 'structural') {
        balanced_score <- (((weighted_cas/100+epsilon)*(simple_cas/100+epsilon)*(mean_mcs/100+epsilon)*(rescaled_silhouette+epsilon))^(1/4)) * 100
    } else {
        balanced_score <- (((weighted_cas/100+epsilon)*(simple_cas/100+epsilon)*(mean_mcs/100+epsilon))^(1/3)) * 100
    }

    metrics <- list(
        "weighted_mean_cas" = weighted_cas, "simple_mean_cas" = simple_cas, "mean_mcs" = mean_mcs,
        "rescaled_silhouette_score" = rescaled_silhouette,
        "balanced_score" = balanced_score,
        "n_individual_labels" = if("ctpt_individual_prediction" %in% names(seurat_final@meta.data)) n_distinct(seurat_final$ctpt_individual_prediction) else 0,
        "n_consensus_labels" = if("ctpt_consensus_prediction" %in% names(seurat_final@meta.data)) n_distinct(seurat_final$ctpt_consensus_prediction) else 0
    )
    return(list(metrics = metrics, seurat_final = seurat_final))
}

print_final_report <- function(target_name, params, metrics, winning_strategy) {
    target_title_map <- list(
        'weighted_cas' = "Weighted Mean CAS", 'simple_cas' = "Simple Mean CAS", 'mcs' = "Mean MCS",
        'balanced' = ifelse(ARGS$model_type == 'structural',
                             "Balanced Score (CAS, MCS & Silhouette)",
                             "Balanced Score (CAS & MCS)")
    )
    target_title <- target_title_map[[target_name]]
    
    cat("\n" %+% paste(rep("=", 60), collapse="") %+% "\n")
    cat(sprintf("--- Final Report for %s Optimization ---\n", target_title))
    cat(sprintf("--- (Best result found by '%s' strategy) ---\n", winning_strategy))
    cat("\n--- Optimal Parameters Found ---\n")
    params$n_hvg <- as.integer(params$n_hvg)
    params$n_pcs <- as.integer(params$n_pcs)
    params$n_neighbors <- as.integer(params$n_neighbors)
    for (key in names(params)) {
        cat(sprintf("  - Best %s: %s\n", key, format(params[[key]], digits=3)))
    }
    
    cat("\n--- Final Metrics for Optimal Parameters ---\n")
    score_name <- paste0("Highest ", gsub(" ", "_", tolower(target_title)))
    cat(sprintf("  - %s: %.2f\n", score_name, if(target_name == 'balanced') metrics$balanced_score else metrics[[paste0(target_name,"_pct")]]))

    cat(sprintf("  - Corresponding Weighted Mean CAS: %.2f%%\n", metrics$weighted_mean_cas))
    cat(sprintf("  - Corresponding Simple Mean CAS: %.2f%%\n", metrics$simple_mean_cas))
    cat(sprintf("  - Corresponding Mean MCS: %.2f%%\n", metrics$mean_mcs))
    cat(sprintf("  - Corresponding Rescaled Silhouette Score: %.3f\n", metrics$rescaled_silhouette_score))
    cat(sprintf("  - Final # of individual cell labels: %d\n", metrics$n_individual_labels))
    cat(sprintf("  - Final # of consensus cluster labels: %d\n", metrics$n_consensus_labels))
    cat(paste(rep("=", 60), collapse="") %+% "\n")
}

save_results_to_file <- function(output_path, target_name, params, metrics, winning_strategy) {
    target_title_map <- list(
        'weighted_cas' = "Weighted Mean CAS", 'simple_cas' = "Simple Mean CAS", 'mcs' = "Mean MCS",
        'balanced' = ifelse(ARGS$model_type == 'structural',
                             "Balanced Score (Geometric Mean of CAS, MCS & Silhouette)",
                             "Balanced Score (Geometric Mean of CAS & MCS)")
    )
    target_title <- target_title_map[[target_name]]
    
    lines <- c(
        "--- Bayesian Optimization Results ---",
        paste("Annotation Method: Seurat Cross-Dataset Anchoring"),
        paste("Species:", ARGS$species), # --- [MODIFICATION] ---
        paste("Optimization Model Type:", ARGS$model_type),
        paste("Optimization Target:", target_title),
        paste("Winning Strategy:", winning_strategy),
        paste("Random Seed Used:", RANDOM_SEED),
        "",
        sapply(names(params), function(key) sprintf("Best %s: %s", key, format(params[[key]], digits=4))),
        "",
        sprintf("Highest_balanced_score: %.4f", metrics$balanced_score),
        sprintf("Corresponding_weighted_mean_cas_pct: %.2f", metrics$weighted_mean_cas),
        sprintf("Corresponding_simple_mean_cas_pct: %.2f", metrics$simple_mean_cas),
        sprintf("Corresponding_mean_mcs_pct: %.2f", metrics$mean_mcs),
        sprintf("Corresponding_rescaled_silhouette_score: %.4f", metrics$rescaled_silhouette_score),
        sprintf("Final_n_individual_labels: %d", metrics$n_individual_labels),
        sprintf("Final_n_consensus_labels: %d", metrics$n_consensus_labels)
    )
    writeLines(lines, output_path)
}

generate_yield_csv <- function(results_dict, target_metric, output_dir, output_prefix) {
    cat("\n--- Generating consolidated yield CSV report ---\n")
    all_dfs <- list()

    for (name in names(results_dict)) {
        result <- results_dict[[name]]
        history_df <- as.data.frame(result$History)
        params_df <- history_df %>%
            dplyr::select(n_hvg, n_pcs, n_neighbors, resolution)
        
        if (length(result$trial_metadata) == nrow(params_df)) {
            metadata_df <- bind_rows(result$trial_metadata)
            base_df <- bind_cols(params_df, metadata_df)
        } else {
            cat(sprintf("  [WARNING] Per-trial metadata mismatch for strategy '%s' (%d vs %d). Metric columns will be empty.\n", name, length(result$trial_metadata), nrow(params_df)))
            base_df <- params_df
        }
        
        base_df$yield_score_target <- history_df$Value
        base_df$call_number <- 1:nrow(base_df)
        base_df$strategy <- name
        
        all_dfs[[name]] <- base_df
    }

    if (length(all_dfs) == 0) {
        cat("  [ERROR] No results found to generate CSV. Skipping.\n")
        return()
    }
        
    final_df <- bind_rows(all_dfs)
    
    epsilon <- 1e-6
    if (all(c('weighted_mean_cas', 'simple_mean_cas', 'mean_mcs', 'silhouette_score') %in% names(final_df))) {
        if (ARGS$model_type == 'structural') {
            final_df$balanced_score_gmean <- with(final_df, 
                (((weighted_mean_cas/100+epsilon)*(simple_mean_cas/100+epsilon)*(mean_mcs/100+epsilon)*(silhouette_score+epsilon))^(1/4)) * 100)
        } else {
            final_df$balanced_score_gmean <- with(final_df, 
                (((weighted_mean_cas/100+epsilon)*(simple_mean_cas/100+epsilon)*(mean_mcs/100+epsilon))^(1/3)) * 100)
        }
    } else {
        final_df$balanced_score_gmean <- NA
    }

    final_column_order <- c(
        'call_number', 'strategy', 'n_hvg', 'n_pcs', 'n_neighbors', 'resolution',
        'yield_score_target', 'balanced_score_gmean', 'weighted_mean_cas', 'simple_mean_cas', 'mean_mcs',
        'silhouette_score', 'n_individual_labels', 'n_consensus_labels'
    )
    final_column_order <- intersect(final_column_order, names(final_df))
    final_df <- final_df[, final_column_order]

    output_path <- file.path(output_dir, paste0(output_prefix, "_", target_metric, "_yield_scores_report.csv"))
    write.csv(final_df, output_path, row.names = FALSE)
    cat(sprintf("✅ Success! Saved consolidated CSV report to: %s\n", output_path))
}

plot_optimizer_paths_tsne <- function(results, target_metric, output_dir, output_prefix, n_points_to_show=25) {
    cat("\n--- Generating t-SNE visualization with publication-quality style ---\n")
    all_points_df <- bind_rows(lapply(results, function(res) as.data.frame(res$History)[, c("n_hvg", "n_pcs", "n_neighbors", "resolution")])) %>% distinct()
    
    min_points_required <- 15 
    if (nrow(all_points_df) < min_points_required) {
        cat(sprintf("     [INFO] Skipping t-SNE plot: only %d unique points found, which is less than the required minimum of %d for a stable embedding.\n",
                nrow(all_points_df), min_points_required))
        return(NULL) 
    }

    perplexity_val <- max(2, min(30, floor((nrow(all_points_df) - 1) / 3)))

    if (nrow(all_points_df) <= 3 * perplexity_val) {
        cat(sprintf("     [INFO] Skipping t-SNE plot: Perplexity of %d is too high for %d unique points. Increase n_calls.\n", perplexity_val, nrow(all_points_df)))
        return(NULL)
    }

    cat(sprintf("Found %d unique points. Performing t-SNE embedding with perplexity=%d...\n",
            nrow(all_points_df), perplexity_val))

    tsne_out <- Rtsne(
        scale(all_points_df),
        perplexity = perplexity_val,
        check_duplicates = FALSE,
        pca_scale = TRUE
    )
    
    tsne_coords <- as.data.frame(tsne_out$Y)
    names(tsne_coords) <- c("TSNE1", "TSNE2")
    
    tsne_map_df <- bind_cols(all_points_df, tsne_coords)

    path_data_list <- list()
    for (name in names(results)) {
        res_history <- as.data.frame(results[[name]]$History)[, c("n_hvg", "n_pcs", "n_neighbors", "resolution")]
        res_history$strategy <- name
        res_history$call_number <- 1:nrow(res_history)
        
        path_data <- res_history %>%
            head(n_points_to_show) %>%
            inner_join(tsne_map_df, by = c("n_hvg", "n_pcs", "n_neighbors", "resolution")) %>%
            arrange(call_number)
        
        path_data <- path_data %>%
            mutate(TSNE1_end = lead(TSNE1), TSNE2_end = lead(TSNE2))
        
        path_data_list[[name]] <- path_data
    }
    path_data_df <- bind_rows(path_data_list)

    colors <- c('Exploit' = '#d62728', 'BO-EI' = "#fcbe06", 'Explore' = "#9015d2")

    p <- ggplot() +
        geom_point(data = tsne_coords, aes(x = TSNE1, y = TSNE2), color = "grey80", size = 4, alpha = 0.5) +
        geom_segment(data = path_data_df, 
                     aes(x = TSNE1, y = TSNE2, xend = TSNE1_end, yend = TSNE2_end, color = strategy),
                     arrow = arrow(length = unit(0.3, "cm"), type = "closed"), size = 1.2) +
        scale_color_manual(values = colors, name = "Strategy") +
        labs(
            title = paste0("Optimizer Paths (First ", n_points_to_show, " Steps)"),
            subtitle = paste("Target:", gsub("_", " ", tools::toTitleCase(target_metric))),
            x = "t-SNE 1", y = "t-SNE 2"
        ) +
        theme_minimal(base_size = 20) +
        theme(
            plot.title = element_text(size = 28, face = "bold"),
            plot.subtitle = element_text(size = 24),
            axis.title = element_text(size = 28, face = "bold"),
            axis.text = element_text(size = 28, face = "bold"),
            legend.title = element_text(size = 28, face = "bold"),
            legend.text = element_text(size = 28, face = "bold"),
            panel.grid = element_blank()
        )
    
    output_path <- file.path(output_dir, paste0(output_prefix, "_", target_metric, "_optimizer_paths_tsne.png"))
    ggsave(output_path, p, width = 12, height = 10, dpi = 500)
    cat(sprintf("✅ Success! Saved t-SNE plot to: %s\n", output_path))
}

plot_optimizer_convergence <- function(results, target_metric, output_dir, output_prefix) {
    cat("\n--- Generating convergence plot with publication-quality style ---\n")
    
    convergence_data <- bind_rows(lapply(names(results), function(name) {
        data.frame(
            call_number = 1:nrow(results[[name]]$History),
            score = results[[name]]$History$Value,
            strategy = name
        )
    }))

    best_so_far <- convergence_data %>%
        group_by(strategy) %>%
        arrange(call_number) %>%
        mutate(best_score = cummax(score))

    colors <- c('Exploit' = '#d62728', 'BO-EI' = "#fcbe06", 'Explore' = "#9015d2")
    
    title_map <- list('weighted_cas' = 'Weighted Mean CAS', 'simple_cas' = 'Simple Mean CAS', 'mcs' = 'Mean MCS',
                 'balanced' = ifelse(ARGS$model_type == 'structural', 
                                     'Balanced Score (CAS, MCS & Silhouette)', 
                                     'Balanced Score (CAS & MCS)'))
    
    p <- ggplot(best_so_far, aes(x = call_number, y = best_score, color = strategy, group = strategy)) +
        geom_line(size = 1.5) +
        geom_point(size = 3) +
        scale_color_manual(values = colors, name = "Strategy") +
        labs(
            title = "Bayesian Optimization Convergence",
            subtitle = paste("Target:", title_map[[target_metric]]),
            x = "Call Number (Experiment Iteration)",
            y = "Best Score Found"
        ) +
        theme_minimal(base_size = 20) +
        theme(
            plot.title = element_text(size = 28, face = "bold"),
            plot.subtitle = element_text(size = 24),
            axis.title = element_text(size = 28, face = "bold"),
            axis.text = element_text(size = 28, face = "bold"),
            legend.title = element_text(size = 28, face = "bold"),
            legend.text = element_text(size = 28, face = "bold"),
            panel.grid.major = element_line(color="gray90"),
            panel.grid.minor = element_blank()
        )

    output_path <- file.path(output_dir, paste0(output_prefix, "_", target_metric, "_optimizer_convergence.png"))
    ggsave(output_path, p, width = 22, height = 10, dpi = 500)
    cat(sprintf("✅ Success! Saved convergence plot to: %s\n", output_path))
}


# ==============================================================================
# --- *** MAIN SCRIPT LOGIC *** ---
# ==============================================================================
main <- function(args) {
    .GlobalEnv$ARGS <- args
    .GlobalEnv$RANDOM_SEED <- args$seed
    .GlobalEnv$OPTIMIZATION_CACHE <- new.env(hash = TRUE)
    set.seed(RANDOM_SEED)
    dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)

    # --- [MODIFICATION START] --- Species-specific setup ---
    cat(sprintf("\n--- Setting up for '%s' species ---\n", args$species))
    if (args$species == "human") {
        check_and_load_bioc_package("org.Hs.eg.db")
        .GlobalEnv$SPECIES_DB <- org.Hs.eg.db
        .GlobalEnv$MITO_PREFIX <- "MT-" # Official prefix for human
    } else if (args$species == "mouse") {
        check_and_load_bioc_package("org.Mm.eg.db")
        .GlobalEnv$SPECIES_DB <- org.Mm.eg.db
        .GlobalEnv$MITO_PREFIX <- "mt-" # Official prefix for mouse
    } else {
        stop("Invalid species specified. This should have been caught by argparse.")
    }
    cat(sprintf("   -> Using annotation database: %s\n", class(.GlobalEnv$SPECIES_DB)[1]))
    cat(sprintf("   -> Using mitochondrial gene prefix: '%s'\n", .GlobalEnv$MITO_PREFIX))
    # --- [MODIFICATION END] ---

    # --- Reference Object Loading (Unchanged) ---
    cat("\n--- Loading and Preprocessing Reference Seurat Object ---\n")
    tryCatch({
        seurat_ref_obj <- readRDS(args$reference_path)
        seurat_ref_obj <- NormalizeData(seurat_ref_obj, verbose = FALSE)
        seurat_ref_obj <- FindVariableFeatures(seurat_ref_obj, method = "vst", nfeatures = 2000, verbose = FALSE)
        seurat_ref_obj <- ScaleData(seurat_ref_obj, verbose = FALSE)
        seurat_ref_obj <- RunPCA(seurat_ref_obj, npcs = 105, verbose = FALSE)
        
        .GlobalEnv$seurat_ref <- seurat_ref_obj
        .GlobalEnv$REF_LABELS_COL <- args$reference_labels_col
        
        if (!(.GlobalEnv$REF_LABELS_COL %in% colnames(.GlobalEnv$seurat_ref@meta.data))) {
            stop(sprintf("The specified reference label column '%s' was not found in the reference object's metadata.", .GlobalEnv$REF_LABELS_COL))
        }

        cat(sprintf("✅ Reference object loaded from '%s' and processed successfully.\n", args$reference_path))
        cat(sprintf("   Using '%s' as the reference labels column.\n", .GlobalEnv$REF_LABELS_COL))
    }, error = function(e) {
        stop(sprintf("Failed to load or process the reference Seurat object from '%s'. Error: %s", args$reference_path, e$message))
    })
    
    # --- [MODIFIED] Initial Data Loading, Gene Name Mapping, and QC ---
    cat("\n--- PART 1: Initial Data Loading and Preprocessing (Seurat) ---\n")
    
    data <- Read10X(data.dir = args$data_dir)
    seurat_obj <- CreateSeuratObject(counts = data, project = "scRNA", min.cells = MIN_CELLS_PER_GENE)

    ensembl_ids_with_version <- rownames(seurat_obj)
    ensembl_ids <- gsub("\\..*$", "", ensembl_ids_with_version)
    
    # --- [MODIFICATION] Use the dynamically set species database ---
    gene_symbols <- mapIds(.GlobalEnv$SPECIES_DB, keys = ensembl_ids, column = "SYMBOL", keytype = "ENSEMBL", multiVals = "first")
    
    unmapped_indices <- which(is.na(gene_symbols))
    gene_symbols[unmapped_indices] <- ensembl_ids_with_version[unmapped_indices]
    unique_gene_symbols <- make.unique(as.character(gene_symbols))
    counts_data <- GetAssayData(seurat_obj, assay = "RNA", layer = "counts")
    rownames(counts_data) <- unique_gene_symbols
    seurat_obj[["RNA"]] <- CreateAssayObject(counts = counts_data)
    DefaultAssay(seurat_obj) <- "RNA"
    
    # --- [MODIFICATION] Use the dynamically set mitochondrial prefix ---
    seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = paste0("^", .GlobalEnv$MITO_PREFIX))
    
    seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > MIN_GENES_PER_CELL & nFeature_RNA < MAX_GENES_PER_CELL & percent.mt < MAX_PCT_COUNTS_MT)
    seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
    
    # --- Subsampling logic (Unchanged from previous version) ---
    .GlobalEnv$seurat_full_data <- seurat_obj
    cat(sprintf("✅ QC complete. FULL dataset contains %d genes and %d cells.\n", nrow(seurat_full_data), ncol(seurat_full_data)))

    seurat_for_opt <- .GlobalEnv$seurat_full_data

    if (!is.null(args$subsample_n_cells) && args$subsample_n_cells > 0) {
        n_total_cells <- ncol(.GlobalEnv$seurat_full_data)
        n_subsample <- min(args$subsample_n_cells, n_total_cells)

        cat(sprintf("\n--- Subsampling data for faster optimization ---\n"))
        if (n_subsample < n_total_cells) {
            cat(sprintf("       -> Will use a random subset of %d out of %d cells for each optimization trial.\n", n_subsample, n_total_cells))
            
            set.seed(RANDOM_SEED)
            subsample_cell_barcodes <- sample(colnames(.GlobalEnv$seurat_full_data), n_subsample)
            seurat_for_opt <- subset(.GlobalEnv$seurat_full_data, cells = subsample_cell_barcodes)
            
            cat(sprintf("✅ Subsampled object for optimization now has %d cells.\n", ncol(seurat_for_opt)))
            cat(sprintf("       -> The FINAL report will still use the FULL dataset of %d cells.\n", n_total_cells))
        } else {
             cat(sprintf("       -> Subsample size (%d) is >= total cells (%d). Using full dataset for optimization.\n", n_subsample, n_total_cells))
        }
    } else {
        cat("\n--- Using FULL dataset for optimization trials (no subsampling) ---\n")
    }
    
    .GlobalEnv$seurat_base <- seurat_for_opt
    
    # --- The rest of the script continues as before (Unchanged) ---
    is_two_step_hvg <- !is.null(args$hvg_min_mean) && !is.null(args$hvg_max_mean) && !is.null(args$hvg_min_disp)

    if (is_two_step_hvg) {
        cat("\n--- Defining candidate gene pool (Two-Step HVG Mode) ---\n")
        seurat_obj_temp <- FindVariableFeatures(seurat_base, method = "vst", nfeatures = nrow(seurat_base), verbose = FALSE)
        hvg_info <- HVFInfo(seurat_obj_temp, method = "vst", assay = "RNA")

        filtered_genes <- subset(hvg_info,
                                 mean > args$hvg_min_mean &
                                 mean < args$hvg_max_mean &
                                 variance.standardized > args$hvg_min_disp)
        
        n_filtered_genes <- nrow(filtered_genes)
        
        cat(sprintf("       -> Common Pool: Found %d genes passing thresholds. This is the candidate pool for optimization.\n", n_filtered_genes))

        original_min_hvg <- search_space_bounds$n_hvg[1]
        
        if (n_filtered_genes < original_min_hvg) {
            stop(sprintf("HVG filtering resulted in only %d genes, which is below the minimum search bound of %d.", n_filtered_genes, original_min_hvg))
        }
        
        cat(sprintf("       -> Adjusting 'n_hvg' search space to [%d, %d].\n", original_min_hvg, n_filtered_genes))
        search_space_bounds$n_hvg[2] <<- as.integer(n_filtered_genes)
    } else {
        cat("\n--- Defining candidate gene pool (Standard HVG Mode) ---\n")
        cat(sprintf("       -> The optimizer will select HVGs from the total pool of %d QC-passed genes.\n", nrow(seurat_base)))
    }
    
    targets_to_run <- if (args$target == 'all') c('balanced') else c(args$target)

    for (target in targets_to_run) {
        target_name_map <- list(
            'weighted_cas' = 'WEIGHTED MEAN CAS', 'simple_cas' = 'SIMPLE MEAN CAS', 'mcs' = 'MEAN MCS',
            'balanced' = ifelse(args$model_type == 'structural',
                                'BALANCED SCORE (CAS, MCS & SILHOUETTE)',
                                'BALANCED SCORE (CAS & MCS)')
        )
        
        cat("\n\n" %+% paste(rep("#", 70), collapse="") %+% "\n")
        cat(sprintf("### STAGE: OPTIMIZING FOR %s ###\n", target_name_map[[target]]))
        cat(sprintf("### Using '%s' Optimization Model ###\n", toupper(args$model_type)))
        cat(sprintf("### Comparing 3 acquisition strategies w/ %d calls each ###\n", args$n_calls))
        cat(paste(rep("#", 70), collapse="") %+% "\n")
        
        .GlobalEnv$CURRENT_OPTIMIZATION_TARGET <- target
        
        strategies <- list(
            "Exploit" = list(acq = 'poi', kappa = 2.576, eps = 0.0),
            "BO-EI"   = list(acq = 'ei',  kappa = 2.576, eps = 0.0),
            "Explore" = list(acq = 'ei',  kappa = 2.576, eps = 0.1)
        )
        
        output_prefix_model <- paste(args$output_prefix, args$model_type, sep = "_")

        results <- list()
        for (name in names(strategies)) {
            cat(sprintf("\n--- Running Strategy: %s ---\n", name))
            .GlobalEnv$CURRENT_STRATEGY_NAME <- name
            .GlobalEnv$TRIAL_METADATA <- list()

            opt_result <- BayesianOptimization(
                FUN = objective_function,
                bounds = search_space_bounds,
                init_points = 5,
                n_iter = args$n_calls - 5,
                acq = strategies[[name]]$acq,
                kappa = strategies[[name]]$kappa,
                eps = strategies[[name]]$eps,
                verbose = FALSE
            )
            
            opt_result$trial_metadata <- .GlobalEnv$TRIAL_METADATA
            results[[name]] <- opt_result
            
            result_path <- file.path(args$output_dir, paste0(output_prefix_model, "_", target, "_", tolower(name), "_opt_result.rds"))
            saveRDS(opt_result, result_path)
            cat(sprintf("Saved %s optimization state to %s\n", name, result_path))
        }
        
        generate_yield_csv(results, target, args$output_dir, output_prefix_model)
        plot_optimizer_paths_tsne(results, target, args$output_dir, output_prefix_model, n_points_to_show = args$n_calls)
        plot_optimizer_convergence(results, target, args$output_dir, output_prefix_model)
        
        best_overall_score <- -Inf
        best_params <- NULL
        winning_strategy_name <- ""
        
        for (name in names(results)) {
            if (results[[name]]$Best_Value > best_overall_score) {
                best_overall_score <- results[[name]]$Best_Value
                best_params <- as.list(results[[name]]$Best_Par)
                winning_strategy_name <- name
            }
        }
        
        cat(sprintf("\n--- Analysis Complete for %s ---\n", target_name_map[[target]]))
        cat(sprintf("Overall best score (%.2f) was found by the '%s' strategy.\n", best_overall_score, winning_strategy_name))
        
        final_run <- evaluate_final_metrics(best_params)
        print_final_report(target, best_params, final_run$metrics, winning_strategy_name)
        
        txt_path <- file.path(args$output_dir, paste0(output_prefix_model, "_", target, "_FINAL_best_params.txt"))
        rds_path <- file.path(args$output_dir, paste0(output_prefix_model, "_", target, "_FINAL_annotated.rds"))
        
        save_results_to_file(txt_path, target, best_params, final_run$metrics, winning_strategy_name)
        saveRDS(final_run$seurat_final, rds_path)
        cat(sprintf("\nFinal optimized results for %s saved to:\n  - %s\n  - %s\n", target, txt_path, rds_path))
    }
    
    cat("\n\n--- All specified optimization pipelines finished successfully! ---\n")
}

# --- [MODIFIED] Argument Parser ---
if (sys.nframe() == 0) {
    parser <- ArgumentParser(description="Run Bayesian Optimization for scRNA-seq analysis (R/Seurat version) using cross-dataset anchoring.")
    
    # --- [MODIFICATION START] --- New argument for species ---
    parser$add_argument("--species", type="character", default="human", choices=c("human", "mouse"),
                        help="Species of the sample data. Determines which annotation database and mitochondrial gene prefix to use. Choices: 'human', 'mouse'.")
    # --- [MODIFICATION END] ---
    
    parser$add_argument("--data_dir", type="character", required=TRUE, help="Path to 10x Genomics data directory for the query dataset.")
    parser$add_argument("--output_dir", type="character", required=TRUE, help="Path for output files.")
    parser$add_argument("--reference_path", type="character", required=TRUE, help="Path to reference Seurat object (.rds).")
    parser$add_argument("--reference_assay", type="character", default="RNA", help="Assay in the reference object to use for anchoring.")
    parser$add_argument("--reference_labels_col", type="character", required=TRUE, help="Metadata column in the reference object containing cell type labels.")
    parser$add_argument("--output_prefix", type="character", default="bayesian_opt", help="Base prefix for output files.")
    parser$add_argument("--seed", type="integer", default=42, help="Random seed for reproducibility.")
    parser$add_argument("--n_calls", type="integer", default=50, help="Number of calls for EACH strategy.")
    parser$add_argument("--subsample_n_cells", type="integer", default=NULL, help="(Optional & Recommended) Number of cells to subsample for fast optimization trials. Final run uses all cells.")
    
    parser$add_argument("--model_type", type="character", default="structural", choices=c("biological", "structural"),
                        help="Select the optimization model: 'biological' or 'structural'.")
    parser$add_argument("--target", type="character", default="all", choices=c("all", "weighted_cas", "simple_cas", "mcs"),
                        help="Specify the optimization target. 'all' defaults to just running the 'balanced' target.")
    
    parser$add_argument("--hvg_min_mean", type="double", default=NULL, help="(Optional) Activates two-step HVG selection. Min mean expression (vst.mean) for initial filtering.")
    parser$add_argument("--hvg_max_mean", type="double", default=NULL, help="(Optional) Activates two-step HVG selection. Max mean expression (vst.mean) for initial filtering.")
    parser$add_argument("--hvg_min_disp", type="double", default=NULL, help="(Optional) Activates two-step HVG selection. Min dispersion (vst.variance.standardized) for initial filtering.")

    parsed_args <- parser$parse_args()
    
    `%+%` <- function(a, b) paste0(a, b)
    
    main(parsed_args)
}