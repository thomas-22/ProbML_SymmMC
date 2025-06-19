# 05_symmetry.R
# ──────────────────────────────────────────────────────────────────────────────
# Tools for parameter set canonicalization to ensure one dei mcmc chain per symmetry group

library(keras)
if (!requireNamespace("proxy", quietly = TRUE)) {
  install.packages(
    "proxy"
  )
}
library(proxy)


# sign-flip for one dense layer
canonical_sign_flip <- function(W, b) {
  scores <- colSums(W) + b
  flips  <- ifelse(scores < 0, -1, 1)
  W_std  <- sweep(W, 2, flips, `*`)
  b_std  <- b * flips
  list(W = W_std, b = b_std)
}

# permutation for one dense layer
canonical_permutation <- function(W, b, W_next = NULL) {
  scores <- sqrt(colSums(W^2) + b^2)
  idx    <- order(scores)
  W_p    <- W[, idx, drop = FALSE]
  b_p    <- b[idx]
  Wn_p   <- if (!is.null(W_next)) W_next[idx, , drop = FALSE] else NULL
  list(W = W_p, b = b_p, W_next = Wn_p)
}

# apply canonicalization and save model with "_canon" suffix
canonicalize_model <- function(model_path) {
  model   <- load_model_tf(model_path)
  w       <- get_weights(model)
  n_layers <- length(w) / 2
  kernels <- w[seq(1, by = 2, length.out = n_layers)]
  biases  <- w[seq(2, by = 2, length.out = n_layers)]
  
  # sign-flip on hidden layers
  for (i in seq_len(n_layers - 1)) {
    sf <- canonical_sign_flip(kernels[[i]], biases[[i]])
    kernels[[i]] <- sf$W
    biases[[i]]  <- sf$b
  }
  
  # permutation on hidden layers
  for (i in seq_len(n_layers - 1)) {
    cp <- canonical_permutation(kernels[[i]], biases[[i]], kernels[[i + 1]])
    kernels[[i]]     <- cp$W
    biases[[i]]      <- cp$b
    kernels[[i + 1]] <- cp$W_next
  }
  
  # set and save
  new_w <- unlist(Map(function(K, B) list(K, B), kernels, biases), recursive = FALSE)
  set_weights(model, new_w)
  dir  <- dirname(model_path)
  base <- basename(model_path)
  out  <- file.path(dir, sub("\\.keras$", "_canon.keras", base))
  save_model_tf(model, out)
  invisible(out)
}

cluster_canonical_models <- function(
    canon_paths,         # character vector of *_canon.keras paths
    threshold    = 1e-6,  # distance threshold for clustering
    metric       = "euclidean",  # distance metric: "euclidean", "cosine", etc.
    output_file  = "canon_reps.txt"
) {
  # Helper: load and flatten model weights
  flatten_weights <- function(path) {
    model   <- load_model_tf(path)
    weights <- get_weights(model)
    as.numeric(unlist(weights))
  }
  
  # Build matrix of flat weight vectors
  mats <- lapply(canon_paths, flatten_weights)
  mat  <- do.call(rbind, mats)
  
  # Compute pairwise distance matrix using chosen metric
  # proxy::dist supports method names like "euclidean","cosine","manhattan"...
  dist_mat <- as.matrix(dist(mat, method = metric))
  
  # Hierarchical clustering (single linkage)
  hc   <- hclust(as.dist(dist_mat), method = "single")
  labs <- cutree(hc, h = threshold)
  
  # Determine representative per cluster
  reps_idx <- tapply(seq_along(labs), labs, function(ix) ix[1])
  reps     <- canon_paths[unlist(reps_idx)]
  
  # Compute centroids of clusters
  unique_labels <- sort(unique(labs))
  centroids <- sapply(unique_labels, function(cl) {
    colMeans(mat[labs == cl, , drop = FALSE])
  })
  centroids <- t(centroids)
  
  # Compute distances between cluster centroids
  centroid_dist <- as.matrix(dist(centroids, method = metric))
  
  # Report chosen metric and distances
  message(sprintf("Using %s distance metric", metric))
  message("Pairwise centroid distances:")
  print(round(centroid_dist, 4))
  
  # Write representative paths to file
  writeLines(reps, con = output_file)
  message("Representatives written to: ", output_file)
  
  # Return reps and centroid distances invisibly
  invisible(list(
    representatives     = reps,
    centroid_distances  = centroid_dist
  ))
}