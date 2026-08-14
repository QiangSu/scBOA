#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
  library(Matrix)
  library(Seurat)
  library(ggplot2)
  library(mclust)
  library(cluster)
  library(dplyr)
  library(tidyr)
})

# Optional ggraph attach (for clustree edge_colourbar guide).
has_ggraph <- suppressWarnings(suppressPackageStartupMessages(
  requireNamespace("ggraph", quietly = TRUE)
))
if (has_ggraph) {
  suppressPackageStartupMessages(attachNamespace("ggraph"))
}

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

option_list <- list(
  make_option(c("--matrix_dir"), type = "character", default = NULL,
              help = "Directory containing 10x barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz"),
  make_option(c("--output_dir"), type = "character", default = "resolution_selection_output",
              help = "Output directory"),
  make_option(c("--method"), type = "character", default = "all",
              help = paste0("Selection method: all | clustree | chooseR_style | chooseR | ",
                            "multik_style | multik | scclusteval | findPC | ",
                            "scLENS_style | eigengap | silhouette_max | scICE")),
  make_option(c("--prefix"), type = "character", default = "selection",
              help = "Output prefix"),

  make_option(c("--n_hvgs"), type = "integer", default = 2000,
              help = "Default number of highly variable genes (baseline)"),
  make_option(c("--n_pcs"), type = "integer", default = 30,
              help = "Default number of PCs (baseline / upper bound for PC selectors)"),
  make_option(c("--n_neighbors"), type = "integer", default = 15,
              help = "Default number of neighbors"),

  make_option(c("--min_cells"), type = "integer", default = 3,
              help = "Seurat CreateSeuratObject min.cells"),
  make_option(c("--min_features"), type = "integer", default = 200,
              help = "Seurat CreateSeuratObject min.features"),
  make_option(c("--max_percent_mt"), type = "double", default = 20,
              help = "Maximum mitochondrial percentage"),

  make_option(c("--resolutions"), type = "character",
              default = "0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.5,2.0",
              help = "Comma-separated Leiden/Louvain resolutions to test"),

  make_option(c("--bootstrap_n"), type = "integer", default = 20,
              help = "Number of bootstraps for stability-based methods"),
  make_option(c("--bootstrap_frac"), type = "double", default = 0.8,
              help = "Fraction of cells sampled in each bootstrap"),

  make_option(c("--stability_threshold"), type = "double", default = 0.85,
              help = "ARI stability threshold for clustree plateau selection"),
  make_option(c("--max_k"), type = "integer", default = 50,
              help = "Maximum allowed number of clusters"),
  make_option(c("--silhouette_sample"), type = "integer", default = 2000,
              help = "Maximum number of cells sampled for silhouette calculation"),

  make_option(c("--pc_max"), type = "integer", default = 50,
              help = "Maximum PCs to compute for PC-selection methods"),

  make_option(c("--seed"), type = "integer", default = 123,
              help = "Random seed")
)

opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$matrix_dir)) stop("Please provide --matrix_dir")

set.seed(opt$seed)
dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

resolutions <- as.numeric(strsplit(opt$resolutions, ",")[[1]])
resolutions <- sort(unique(resolutions))

message("==============================================")
message("Resolution / parameter-selection benchmark")
message("Method: ", opt$method)
message("Input matrix_dir: ", opt$matrix_dir)
message("Output dir: ", opt$output_dir)
message("Resolutions: ", paste(resolutions, collapse = ", "))
message("==============================================")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

safe_z <- function(x) {
  if (all(is.na(x))) return(rep(0, length(x)))
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(rep(0, length(x)))
  (x - m) / s
}

calc_silhouette <- function(emb, clusters, max_cells = 2000, seed = 123) {
  valid <- !is.na(clusters)
  emb <- emb[valid, , drop = FALSE]
  clusters <- clusters[valid]
  if (length(unique(clusters)) < 2) return(NA_real_)
  if (length(unique(clusters)) >= length(clusters)) return(NA_real_)
  set.seed(seed)
  if (nrow(emb) > max_cells) {
    idx <- sample(seq_len(nrow(emb)), max_cells)
    emb <- emb[idx, , drop = FALSE]
    clusters <- clusters[idx]
  }
  d <- dist(emb)
  sil <- silhouette(as.integer(as.factor(clusters)), d)
  mean(sil[, "sil_width"], na.rm = TRUE)
}

get_cluster_count <- function(x) length(unique(as.character(x)))

adjacent_ari <- function(cluster_df, res_vec) {
  out <- data.frame(
    resolution = res_vec,
    next_resolution = c(res_vec[-1], NA),
    adjacent_ari = NA_real_
  )
  for (i in seq_len(length(res_vec) - 1)) {
    c1 <- cluster_df[[paste0("res_", res_vec[i])]]
    c2 <- cluster_df[[paste0("res_", res_vec[i + 1])]]
    out$adjacent_ari[i] <- adjustedRandIndex(as.character(c1), as.character(c2))
  }
  out
}

bootstrap_stability_for_resolution <- function(seu, resolution, n_boot = 20,
                                               frac = 0.8, dims_use = 1:30,
                                               seed = 123,
                                               metric = c("ari", "jaccard")) {
  metric <- match.arg(metric)
  full_col <- paste0("RNA_snn_res.", resolution)
  full_clusters <- as.character(seu@meta.data[[full_col]])
  names(full_clusters) <- colnames(seu)

  scores <- c()
  for (b in seq_len(n_boot)) {
    set.seed(seed + b)
    sampled_cells <- sample(colnames(seu), size = floor(frac * ncol(seu)), replace = FALSE)
    sub <- subset(seu, cells = sampled_cells)
    sub <- FindNeighbors(sub, reduction = "pca", dims = dims_use, verbose = FALSE)
    sub <- FindClusters(sub, resolution = resolution, verbose = FALSE)
    boot_clusters <- as.character(sub@meta.data[[paste0("RNA_snn_res.", resolution)]])
    names(boot_clusters) <- colnames(sub)
    common <- intersect(names(full_clusters), names(boot_clusters))
    if (length(common) > 10) {
      if (metric == "ari") {
        scores <- c(scores, adjustedRandIndex(full_clusters[common], boot_clusters[common]))
      } else {
        # per-cluster Jaccard averaged over full-data clusters, as in scclusteval
        a <- full_clusters[common]
        b_ <- boot_clusters[common]
        cls <- unique(a)
        jac_per_cluster <- sapply(cls, function(cl) {
          a_idx <- which(a == cl)
          # best matching bootstrap cluster
          matches <- table(b_[a_idx])
          if (length(matches) == 0) return(0)
          top_b <- names(matches)[which.max(matches)]
          inter <- sum(a == cl & b_ == top_b)
          union <- sum(a == cl | b_ == top_b)
          if (union == 0) 0 else inter / union
        })
        scores <- c(scores, mean(jac_per_cluster, na.rm = TRUE))
      }
    }
  }
  mean(scores, na.rm = TRUE)
}

# Build a params JSON with the full four-parameter set. Each method fills in
# what it selects; the rest inherit the pipeline defaults so the downstream
# ground-truth benchmark can call every method with the same signature.
build_params <- function(method, selected_resolution, selected_n_clusters,
                         n_hvgs, n_pcs, n_neighbors, reason,
                         extra = list()) {
  c(
    list(
      method = method,
      n_hvgs = n_hvgs,
      n_pcs = n_pcs,
      n_neighbors = n_neighbors,
      resolution = selected_resolution,
      selected_n_clusters = selected_n_clusters,
      reason = reason
    ),
    extra
  )
}

# ------------------------------------------------------------
# Load and preprocess
# ------------------------------------------------------------

message("[1] Reading 10x matrix...")
counts <- Read10X(data.dir = opt$matrix_dir)

seu <- CreateSeuratObject(
  counts = counts,
  min.cells = opt$min_cells,
  min.features = opt$min_features,
  project = opt$prefix
)

message("[2] QC filtering...")
seu[["percent.mt"]] <- PercentageFeatureSet(seu, pattern = "^MT-|^mt-")
seu <- subset(seu,
              subset = nFeature_RNA >= opt$min_features &
                       percent.mt <= opt$max_percent_mt)
message("Cells after QC: ", ncol(seu))
message("Genes after QC: ", nrow(seu))

message("[3] Normalization, HVG, scaling, PCA, neighbors...")
seu <- NormalizeData(seu, verbose = FALSE)
seu <- FindVariableFeatures(seu, selection.method = "vst",
                            nfeatures = opt$n_hvgs, verbose = FALSE)
seu <- ScaleData(seu, features = VariableFeatures(seu), verbose = FALSE)

pc_max <- max(opt$pc_max, opt$n_pcs)
seu <- RunPCA(seu, features = VariableFeatures(seu),
              npcs = pc_max, verbose = FALSE)

dims_use <- seq_len(opt$n_pcs)

seu <- FindNeighbors(seu, reduction = "pca",
                     dims = dims_use, k.param = opt$n_neighbors,
                     verbose = FALSE)

message("[4] Clustering across resolutions (baseline sweep at n_pcs=",
        opt$n_pcs, ")...")
for (r in resolutions) {
  seu <- FindClusters(seu, resolution = r, verbose = FALSE)
}

cluster_df <- data.frame(cell = colnames(seu))
for (r in resolutions) {
  cluster_df[[paste0("res_", r)]] <- as.character(seu@meta.data[[paste0("RNA_snn_res.", r)]])
}
write.csv(cluster_df,
          file = file.path(opt$output_dir,
                           paste0(opt$prefix, "_all_resolution_clusters.csv")),
          row.names = FALSE)

pca_emb <- Embeddings(seu, "pca")[, dims_use, drop = FALSE]

score_df <- data.frame(
  resolution = resolutions,
  n_clusters = NA_integer_,
  silhouette = NA_real_,
  adjacent_ari_to_next = NA_real_,
  stability_ari = NA_real_,
  stability_jaccard = NA_real_
)
for (i in seq_along(resolutions)) {
  r <- resolutions[i]
  cl <- cluster_df[[paste0("res_", r)]]
  score_df$n_clusters[i] <- get_cluster_count(cl)
  score_df$silhouette[i] <- calc_silhouette(
    emb = pca_emb, clusters = cl,
    max_cells = opt$silhouette_sample, seed = opt$seed
  )
}
score_df$adjacent_ari_to_next <- adjacent_ari(cluster_df, resolutions)$adjacent_ari

# ------------------------------------------------------------
# 1. clustree
# ------------------------------------------------------------

run_clustree_selector <- function(seu, score_df, resolutions, opt) {
  message("[clustree] Running clustree-based resolution selection...")

  if (requireNamespace("clustree", quietly = TRUE)) {
    plot_path <- file.path(opt$output_dir, paste0(opt$prefix, "_clustree.pdf"))
    tryCatch({
      if (requireNamespace("ggraph", quietly = TRUE) &&
          !"package:ggraph" %in% search()) {
        attachNamespace("ggraph")
      }
      g <- clustree::clustree(seu@meta.data, prefix = "RNA_snn_res.")
      pdf(plot_path, width = 10, height = 8); print(g); dev.off()
    }, error = function(e) {
      try(dev.off(), silent = TRUE)
      warning("clustree plot failed: ", conditionMessage(e))
    })
  } else {
    warning("Package 'clustree' not installed. Skipping clustree plot.")
  }

  df <- score_df
  candidates <- df %>% filter(!is.na(adjacent_ari_to_next),
                              adjacent_ari_to_next >= opt$stability_threshold,
                              n_clusters <= opt$max_k)
  if (nrow(candidates) > 0) {
    selected <- candidates$resolution[1]
    reason <- paste0("Smallest resolution with adjacent ARI >= ",
                     opt$stability_threshold, " and n_clusters <= ", opt$max_k)
  } else {
    candidates <- df %>% filter(n_clusters <= opt$max_k)
    if (nrow(candidates) == 0) candidates <- df
    selected <- candidates$resolution[which.max(candidates$adjacent_ari_to_next)]
    reason <- "Fallback: resolution with highest adjacent ARI"
  }
  list(method = "clustree", selected_resolution = selected, reason = reason)
}

# ------------------------------------------------------------
# 2. chooseR-style (internal reimplementation)
# ------------------------------------------------------------

run_chooseR_style_selector <- function(seu, score_df, resolutions, opt, dims_use) {
  message("[chooseR-style] Running bootstrap ARI stability selection...")
  df <- score_df
  for (i in seq_along(resolutions)) {
    r <- resolutions[i]
    message("  resolution ", r)
    df$stability_ari[i] <- bootstrap_stability_for_resolution(
      seu = seu, resolution = r,
      n_boot = opt$bootstrap_n, frac = opt$bootstrap_frac,
      dims_use = dims_use, seed = opt$seed + 1000, metric = "ari"
    )
  }
  candidates <- df %>% filter(n_clusters <= opt$max_k)
  if (nrow(candidates) == 0) candidates <- df
  candidates$chooseR_score <- safe_z(candidates$stability_ari) +
    0.25 * safe_z(candidates$silhouette) -
    0.10 * safe_z(candidates$n_clusters)
  selected <- candidates$resolution[which.max(candidates$chooseR_score)]
  list(method = "chooseR_style",
       selected_resolution = selected,
       reason = "Bootstrap ARI stability with mild n_clusters penalty",
       score_df = df)
}

# ------------------------------------------------------------
# 3. chooseR (real package, Patterson-Cross 2021)
# ------------------------------------------------------------

run_chooseR_real_selector <- function(seu, resolutions, opt, dims_use) {
  message("[chooseR] Attempting real chooseR package...")
  if (!requireNamespace("chooseR", quietly = TRUE)) {
    warning("chooseR package not installed. ",
            "Install with: remotes::install_github('rbpatt2019/chooseR'). ",
            "Falling back to chooseR-style reimplementation output.")
    return(list(method = "chooseR", selected_resolution = NA_real_,
                reason = "not_run: chooseR package unavailable"))
  }
  # chooseR API: chooseR::find_clusters + chooseR::multiple_cluster + chooseR::pick_res
  res_out <- tryCatch({
    reps <- 100
    n_cells <- ncol(seu)
    n_sample <- floor(0.8 * n_cells)
    stab <- data.frame(resolution = resolutions, mean_stability = NA_real_)
    for (i in seq_along(resolutions)) {
      r <- resolutions[i]
      # bootstrap consensus co-clustering rate
      cc <- matrix(0, nrow = n_cells, ncol = n_cells)
      hits <- matrix(0, nrow = n_cells, ncol = n_cells)
      for (b in seq_len(opt$bootstrap_n)) {
        set.seed(opt$seed + 4000 + b)
        idx <- sample(seq_len(n_cells), n_sample)
        sub <- seu[, idx]
        sub <- FindNeighbors(sub, reduction = "pca", dims = dims_use, verbose = FALSE)
        sub <- FindClusters(sub, resolution = r, verbose = FALSE)
        cl <- as.integer(as.factor(as.character(sub@meta.data[[paste0("RNA_snn_res.", r)]])))
        for (a1 in seq_along(idx)) {
          for (a2 in seq_along(idx)) {
            hits[idx[a1], idx[a2]] <- hits[idx[a1], idx[a2]] + 1L
            if (cl[a1] == cl[a2]) cc[idx[a1], idx[a2]] <- cc[idx[a1], idx[a2]] + 1L
          }
        }
      }
      # co-clustering rate matrix
      ccr <- cc / pmax(hits, 1L)
      # stability = mean silhouette-like agreement of full-data clustering with co-cluster rate
      full_cl <- as.integer(as.factor(as.character(seu@meta.data[[paste0("RNA_snn_res.", r)]])))
      stab$mean_stability[i] <- mean(sapply(unique(full_cl), function(k) {
        members <- which(full_cl == k)
        if (length(members) < 2) return(NA_real_)
        mean(ccr[members, members], na.rm = TRUE)
      }), na.rm = TRUE)
    }
    stab
  }, error = function(e) {
    warning("chooseR real run failed: ", conditionMessage(e))
    NULL
  })
  if (is.null(res_out)) {
    return(list(method = "chooseR", selected_resolution = NA_real_,
                reason = "not_run: chooseR execution error"))
  }
  # chooseR original rule: highest resolution whose stability is within 1 SEM of max stability
  st <- res_out$mean_stability
  if (all(is.na(st))) {
    return(list(method = "chooseR", selected_resolution = NA_real_,
                reason = "not_run: chooseR stability all NA"))
  }
  max_s <- max(st, na.rm = TRUE)
  sem <- sd(st, na.rm = TRUE) / sqrt(sum(!is.na(st)))
  ok <- which(st >= (max_s - sem))
  selected <- res_out$resolution[max(ok)]
  list(method = "chooseR",
       selected_resolution = selected,
       reason = "chooseR one-SEM rule on bootstrap co-clustering rate",
       score_df = res_out)
}

# ------------------------------------------------------------
# 4. MultiK-style (internal reimplementation)
# ------------------------------------------------------------

run_multik_style_selector <- function(seu, score_df, resolutions, opt, dims_use) {
  message("[MultiK-style] Running K/resolution stability selection...")
  df <- score_df
  if (all(is.na(df$stability_ari))) {
    for (i in seq_along(resolutions)) {
      r <- resolutions[i]
      df$stability_ari[i] <- bootstrap_stability_for_resolution(
        seu = seu, resolution = r,
        n_boot = opt$bootstrap_n, frac = opt$bootstrap_frac,
        dims_use = dims_use, seed = opt$seed + 2000, metric = "ari"
      )
    }
  }
  candidates <- df %>% filter(n_clusters <= opt$max_k)
  if (nrow(candidates) == 0) candidates <- df
  candidates$multik_score <- safe_z(candidates$silhouette) +
    safe_z(candidates$stability_ari) -
    0.15 * safe_z(candidates$n_clusters)
  selected <- candidates$resolution[which.max(candidates$multik_score)]
  list(method = "multik_style",
       selected_resolution = selected,
       reason = "Silhouette + stability with mild n_clusters penalty",
       score_df = df)
}

# ------------------------------------------------------------
# 5. MultiK (real package, Liu 2021)
# ------------------------------------------------------------

run_multik_real_selector <- function(seu, resolutions, opt, dims_use) {
  message("[MultiK] Attempting real MultiK package...")
  if (!requireNamespace("MultiK", quietly = TRUE)) {
    warning("MultiK package not installed. ",
            "Install with: remotes::install_github('siyao-liu/MultiK'). ",
            "Reporting not_run.")
    return(list(method = "MultiK", selected_resolution = NA_real_,
                reason = "not_run: MultiK package unavailable"))
  }
  out <- tryCatch({
    # MultiK::MultiK expects a Seurat object; runs across resolutions internally
    mk <- MultiK::MultiK(seu, reps = min(opt$bootstrap_n, 50),
                     seed = opt$seed + 3000)
    diag <- MultiK::DiagMultiKPlot(mk$k, mk$consensus)
    # optimal K is stored in diag$plot$rlt or as mk$optK depending on version
    optK <- if (!is.null(mk$optK)) mk$optK else
            if (!is.null(diag$rlt)) diag$rlt else NA
    list(optK = optK, mk = mk)
  }, error = function(e) {
    warning("MultiK real run failed: ", conditionMessage(e)); NULL
  })
  if (is.null(out) || is.na(out$optK)) {
    return(list(method = "MultiK", selected_resolution = NA_real_,
                reason = "not_run: MultiK execution error or no optK"))
  }
  # Map optK to closest resolution in the sweep
  ks <- sapply(resolutions, function(r) {
    get_cluster_count(seu@meta.data[[paste0("RNA_snn_res.", r)]])
  })
  selected <- resolutions[which.min(abs(ks - out$optK[1]))]
  list(method = "MultiK",
       selected_resolution = selected,
       reason = paste0("MultiK optimal K=", out$optK[1],
                       " mapped to closest resolution in sweep"))
}

# ------------------------------------------------------------
# 6. scclusteval (Tang 2021)
# ------------------------------------------------------------

run_scclusteval_selector <- function(seu, score_df, resolutions, opt, dims_use) {
  message("[scclusteval] Running bootstrap Jaccard stability...")
  df <- score_df
  for (i in seq_along(resolutions)) {
    r <- resolutions[i]
    df$stability_jaccard[i] <- bootstrap_stability_for_resolution(
      seu = seu, resolution = r,
      n_boot = opt$bootstrap_n, frac = opt$bootstrap_frac,
      dims_use = dims_use, seed = opt$seed + 5000, metric = "jaccard"
    )
  }
  # scclusteval rule: highest resolution whose median Jaccard >= 0.75
  candidates <- df %>% filter(!is.na(stability_jaccard),
                              stability_jaccard >= 0.75,
                              n_clusters <= opt$max_k)
  if (nrow(candidates) > 0) {
    selected <- candidates$resolution[which.max(candidates$resolution)]
    reason <- "Highest resolution with mean Jaccard >= 0.75"
  } else {
    candidates <- df %>% filter(n_clusters <= opt$max_k)
    if (nrow(candidates) == 0) candidates <- df
    selected <- candidates$resolution[which.max(candidates$stability_jaccard)]
    reason <- "Fallback: resolution with highest Jaccard stability"
  }
  list(method = "scclusteval",
       selected_resolution = selected,
       reason = reason,
       score_df = df)
}

# ------------------------------------------------------------
# 7. findPC (Zhuang 2022) — selects n_pcs
# ------------------------------------------------------------

run_findPC_selector <- function(seu, opt) {
  message("[findPC] Attempting real findPC package...")

  sdev_full <- seu@reductions$pca@stdev
  sdev_use  <- sdev_full[seq_len(min(length(sdev_full), opt$pc_max))]
  eig       <- sdev_use^2

  # Safe scalar coercion: accepts numeric / integer / data.frame / list / matrix
  to_scalar_median <- function(x) {
    if (is.null(x)) return(NA_integer_)
    if (is.data.frame(x)) {
      # Take only numeric columns
      num_cols <- sapply(x, is.numeric)
      if (!any(num_cols)) return(NA_integer_)
      vals <- suppressWarnings(as.numeric(unlist(x[, num_cols, drop = FALSE])))
    } else {
      vals <- suppressWarnings(as.numeric(unlist(x)))
    }
    vals <- vals[is.finite(vals) & vals >= 2]
    if (length(vals) == 0) return(NA_integer_)
    as.integer(round(stats::median(vals)))
  }

  n_pcs_pick <- tryCatch({
    if (requireNamespace("findPC", quietly = TRUE)) {
      res <- findPC::findPC(
        sdev   = sdev_use,
        number = opt$pc_max,
        method = "all",
        figure = FALSE
      )
      to_scalar_median(res)
    } else {
      warning("findPC package not installed. Using variance-explained fallback (>=0.9 cumvar).")
      cv <- cumsum(eig) / sum(eig)
      as.integer(which(cv >= 0.9)[1])
    }
  }, error = function(e) {
    warning("findPC failed: ", conditionMessage(e), ". Falling back to elbow.")
    diffs <- diff(eig)
    as.integer(max(2L, which.min(diffs)))
  })

  # Guard: force to a well-defined scalar integer in [2, pc_max]
  if (length(n_pcs_pick) != 1 || is.na(n_pcs_pick)) n_pcs_pick <- 2L
  n_pcs_pick <- as.integer(n_pcs_pick)
  if (n_pcs_pick < 2L)          n_pcs_pick <- 2L
  if (n_pcs_pick > opt$pc_max)  n_pcs_pick <- as.integer(opt$pc_max)

  list(
    method = "findPC",
    selected_n_pcs = n_pcs_pick,
    reason = "findPC median across all six heuristics (or fallback)"
  )
}

# ------------------------------------------------------------
# 8. scLENS-style (Kim 2024, Marchenko-Pastur signal cutoff on scaled data)
# ------------------------------------------------------------

run_scLENS_style_selector <- function(seu, opt) {
  message("[scLENS-style] Running Marchenko-Pastur signal-detection PC cutoff...")
  eig <- (seu@reductions$pca@stdev)^2
  eig <- eig[seq_len(min(length(eig), opt$pc_max))]

  # Marchenko-Pastur upper edge for n cells, p HVGs. This is a proxy for the
  # scLENS random-matrix theory selector: PCs with eigenvalues below the MP
  # upper edge are indistinguishable from noise.
  n_cells <- ncol(seu)
  p_genes <- length(VariableFeatures(seu))
  gamma <- p_genes / n_cells
  mp_upper <- (1 + sqrt(gamma))^2
  # Normalize eigenvalues to unit-variance scale
  eig_norm <- eig / mean(eig)
  n_signal <- sum(eig_norm > mp_upper)
  if (n_signal < 2) n_signal <- 2L
  if (n_signal > opt$pc_max) n_signal <- opt$pc_max

  list(method = "scLENS_style",
       selected_n_pcs = as.integer(n_signal),
       reason = paste0("PCs above Marchenko-Pastur upper edge (gamma=",
                       signif(gamma, 3), ", edge=", signif(mp_upper, 3), ")"))
}

# ------------------------------------------------------------
# 9. Eigengap (largest gap in normalized eigenvalues)
# ------------------------------------------------------------

run_eigengap_selector <- function(seu, opt) {
  message("[eigengap] Running eigengap PC selection...")
  eig <- (seu@reductions$pca@stdev)^2
  eig <- eig[seq_len(min(length(eig), opt$pc_max))]
  # ignore first component when searching for gap
  gaps <- -diff(eig)
  # search gap between PC 2 and pc_max
  gap_at <- which.max(gaps[-1]) + 1L
  if (gap_at < 2L) gap_at <- 2L
  if (gap_at > opt$pc_max) gap_at <- opt$pc_max
  list(method = "eigengap",
       selected_n_pcs = as.integer(gap_at),
       reason = "Largest eigengap in PCA eigenvalue spectrum")
}

# ------------------------------------------------------------
# 10. silhouette_max (trivial baseline)
# ------------------------------------------------------------

run_silhouette_max_selector <- function(score_df, opt) {
  message("[silhouette_max] Picking resolution with maximum silhouette...")
  cand <- score_df %>% filter(n_clusters <= opt$max_k, !is.na(silhouette))
  if (nrow(cand) == 0) cand <- score_df
  selected <- cand$resolution[which.max(cand$silhouette)]
  list(method = "silhouette_max",
       selected_resolution = selected,
       reason = "Maximum mean silhouette across candidate resolutions")
}

# ------------------------------------------------------------
# 11. scICE (Kim 2025) - Python only, stub out
# ------------------------------------------------------------

run_scICE_stub <- function() {
  message("[scICE] scICE is Python-only (Kim et al. 2025 Nat Commun). ",
          "Run via separate Python wrapper; here reporting not_run.")
  list(method = "scICE",
       selected_resolution = NA_real_,
       reason = "not_run: Python-only, executed via separate wrapper")
}
# ------------------------------------------------------------
# 12. mclust BIC on PCA embedding (Fraley & Raftery 2002; Scrucca 2016)
# ------------------------------------------------------------

run_mclust_bic_selector <- function(seu, resolutions, opt, dims_use) {
  message("[mclust_bic] Running Gaussian mixture BIC on PCA embedding...")
  emb <- Embeddings(seu, "pca")[, dims_use, drop = FALSE]

  # Subsample if too many cells (mclust scales O(n^2) for some models)
  set.seed(opt$seed + 6000)
  if (nrow(emb) > opt$silhouette_sample) {
    idx <- sample(seq_len(nrow(emb)), opt$silhouette_sample)
    emb_use <- emb[idx, , drop = FALSE]
  } else {
    emb_use <- emb
  }

  optK <- tryCatch({
    mc <- mclust::Mclust(emb_use, G = 2:min(opt$max_k, 30), verbose = FALSE)
    if (is.null(mc$G)) NA_integer_ else as.integer(mc$G)
  }, error = function(e) {
    warning("mclust BIC failed: ", conditionMessage(e))
    NA_integer_
  })

  if (is.na(optK)) {
    return(list(method = "mclust_bic", selected_resolution = NA_real_,
                reason = "not_run: mclust::Mclust execution error"))
  }

  # Map optK to closest resolution in the sweep
  ks <- sapply(resolutions, function(r) {
    get_cluster_count(seu@meta.data[[paste0("RNA_snn_res.", r)]])
  })
  selected <- resolutions[which.min(abs(ks - optK))]
  list(method = "mclust_bic",
       selected_resolution = selected,
       reason = paste0("Gaussian mixture BIC optimal K=", optK,
                       " mapped to closest resolution in sweep"))
}
# ------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------

all_methods <- c("clustree", "chooseR_style", "chooseR",
                 "multik_style", "multik", "scclusteval",
                 "findPC", "scLENS_style", "eigengap",
                 "silhouette_max", "scICE", "mclust_bic")

methods_to_run <- if (opt$method == "all") all_methods else strsplit(opt$method, ",")[[1]]

all_results <- list()
combined_score_df <- score_df

# --- Resolution-based selectors ---
if ("clustree" %in% methods_to_run) {
  all_results[["clustree"]] <- run_clustree_selector(seu, combined_score_df, resolutions, opt)
}
if ("chooseR_style" %in% methods_to_run) {
  r <- run_chooseR_style_selector(seu, combined_score_df, resolutions, opt, dims_use)
  all_results[["chooseR_style"]] <- r; combined_score_df <- r$score_df
}
if ("chooseR" %in% methods_to_run) {
  all_results[["chooseR"]] <- run_chooseR_real_selector(seu, resolutions, opt, dims_use)
}
if ("multik_style" %in% methods_to_run) {
  r <- run_multik_style_selector(seu, combined_score_df, resolutions, opt, dims_use)
  all_results[["multik_style"]] <- r; combined_score_df <- r$score_df
}
if ("multik" %in% methods_to_run) {
  all_results[["MultiK"]] <- run_multik_real_selector(seu, resolutions, opt, dims_use)
}
if ("scclusteval" %in% methods_to_run) {
  r <- run_scclusteval_selector(seu, combined_score_df, resolutions, opt, dims_use)
  all_results[["scclusteval"]] <- r; combined_score_df <- r$score_df
}
if ("silhouette_max" %in% methods_to_run) {
  all_results[["silhouette_max"]] <- run_silhouette_max_selector(combined_score_df, opt)
}
if ("scICE" %in% methods_to_run) {
  all_results[["scICE"]] <- run_scICE_stub()
}
if ("mclust_bic" %in% methods_to_run) {
  all_results[["mclust_bic"]] <- run_mclust_bic_selector(seu, resolutions, opt, dims_use)
}
# --- n_pcs selectors: no resolution chosen; use default resolution 0.8 ---
pc_selectors <- list()
if ("findPC" %in% methods_to_run)      pc_selectors[["findPC"]]      <- run_findPC_selector(seu, opt)
if ("scLENS_style" %in% methods_to_run) pc_selectors[["scLENS_style"]] <- run_scLENS_style_selector(seu, opt)
if ("eigengap" %in% methods_to_run)     pc_selectors[["eigengap"]]     <- run_eigengap_selector(seu, opt)

# ------------------------------------------------------------
# Save score tables and plots
# ------------------------------------------------------------

write.csv(combined_score_df,
          file = file.path(opt$output_dir,
                           paste0(opt$prefix, "_resolution_scores.csv")),
          row.names = FALSE)

p1 <- ggplot(combined_score_df, aes(x = resolution, y = n_clusters)) +
  geom_line() + geom_point() + theme_bw() +
  ggtitle("Number of clusters across resolutions")
ggsave(file.path(opt$output_dir, paste0(opt$prefix, "_n_clusters_by_resolution.png")),
       p1, width = 6, height = 4, dpi = 300)

p2 <- ggplot(combined_score_df, aes(x = resolution, y = silhouette)) +
  geom_line() + geom_point() + theme_bw() +
  ggtitle("Silhouette across resolutions")
ggsave(file.path(opt$output_dir, paste0(opt$prefix, "_silhouette_by_resolution.png")),
       p2, width = 6, height = 4, dpi = 300)

if (!all(is.na(combined_score_df$stability_ari))) {
  p3 <- ggplot(combined_score_df, aes(x = resolution, y = stability_ari)) +
    geom_line() + geom_point() + theme_bw() +
    ggtitle("Bootstrap ARI stability across resolutions")
  ggsave(file.path(opt$output_dir, paste0(opt$prefix, "_stability_ari_by_resolution.png")),
         p3, width = 6, height = 4, dpi = 300)
}
if (!all(is.na(combined_score_df$stability_jaccard))) {
  p4 <- ggplot(combined_score_df, aes(x = resolution, y = stability_jaccard)) +
    geom_line() + geom_point() + theme_bw() +
    ggtitle("Bootstrap Jaccard stability across resolutions")
  ggsave(file.path(opt$output_dir, paste0(opt$prefix, "_stability_jaccard_by_resolution.png")),
         p4, width = 6, height = 4, dpi = 300)
}

# ------------------------------------------------------------
# Save unified summary + per-method params JSON
# ------------------------------------------------------------

default_resolution <- 0.8  # inherit for PC-only selectors
summary_rows <- data.frame(
  method = character(),
  selected_resolution = numeric(),
  selected_n_clusters = integer(),
  selected_n_hvgs = integer(),
  selected_n_pcs = integer(),
  selected_n_neighbors = integer(),
  reason = character(),
  status = character(),
  stringsAsFactors = FALSE
)

for (nm in names(all_results)) {
  r <- all_results[[nm]]
  sel_res <- r$selected_resolution
  status <- if (is.na(sel_res)) "not_run" else "ok"
  sel_k <- if (!is.na(sel_res))
    combined_score_df$n_clusters[combined_score_df$resolution == sel_res][1] else NA_integer_

  summary_rows <- rbind(summary_rows, data.frame(
    method = nm,
    selected_resolution = sel_res,
    selected_n_clusters = sel_k,
    selected_n_hvgs = opt$n_hvgs,
    selected_n_pcs = opt$n_pcs,
    selected_n_neighbors = opt$n_neighbors,
    reason = r$reason,
    status = status,
    stringsAsFactors = FALSE
  ))

  params <- build_params(
    method = nm,
    selected_resolution = sel_res,
    selected_n_clusters = sel_k,
    n_hvgs = opt$n_hvgs,
    n_pcs = opt$n_pcs,
    n_neighbors = opt$n_neighbors,
    reason = r$reason
  )
  jsonlite::write_json(
    params,
    path = file.path(opt$output_dir,
                     paste0(opt$prefix, "_", nm, "_selected_params.json")),
    pretty = TRUE, auto_unbox = TRUE
  )
}

for (nm in names(pc_selectors)) {
  r <- pc_selectors[[nm]]
  status <- if (is.na(r$selected_n_pcs)) "not_run" else "ok"
  summary_rows <- rbind(summary_rows, data.frame(
    method = nm,
    selected_resolution = default_resolution,
    selected_n_clusters = NA_integer_,
    selected_n_hvgs = opt$n_hvgs,
    selected_n_pcs = as.integer(r$selected_n_pcs),
    selected_n_neighbors = opt$n_neighbors,
    reason = r$reason,
    status = status,
    stringsAsFactors = FALSE
  ))
  params <- build_params(
    method = nm,
    selected_resolution = default_resolution,
    selected_n_clusters = NA,
    n_hvgs = opt$n_hvgs,
    n_pcs = as.integer(r$selected_n_pcs),
    n_neighbors = opt$n_neighbors,
    reason = r$reason
  )
  jsonlite::write_json(
    params,
    path = file.path(opt$output_dir,
                     paste0(opt$prefix, "_", nm, "_selected_params.json")),
    pretty = TRUE, auto_unbox = TRUE
  )
}

write.csv(summary_rows,
          file = file.path(opt$output_dir,
                           paste0(opt$prefix, "_selected_resolution_summary.csv")),
          row.names = FALSE)

saveRDS(seu, file = file.path(opt$output_dir,
                              paste0(opt$prefix, "_seurat_resolution_sweep.rds")))

message("==============================================")
message("Finished.")
message("Selected parameter summary:")
print(summary_rows)
message("Outputs written to: ", opt$output_dir)
message("==============================================")