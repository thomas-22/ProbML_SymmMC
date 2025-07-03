# 03_canon.R
# ──────────────────────────────────────────────────────────────────────────────
# Tools for parameter set canonicalization to ensure one dei mcmc chain per symmetry group

library(keras)
if (!requireNamespace("proxy", quietly = TRUE)) {
  install.packages(
    "proxy"
  )
}
library(proxy)


# 1) Sign-Flip-Standardisierung pro Dense-Layer
#    W: (input_dim × units), b: length units, W_next: (units × units_next)
canonical_sign_flip <- function(W, b, W_next = NULL) {
  # Score für jedes Neuron = Summe der Spalte + Bias
  flips <- ifelse(colSums(W) + b < 0, -1, 1)
  
  # W und b in Layer i standardisieren
  W_std <- sweep(W, 2, flips, `*`)
  b_std <- b * flips
  
  # Wenn vorhanden, auch die Zeilen von W_next invertieren
  if (!is.null(W_next)) {
    Wn_std <- sweep(W_next, 1, flips, `*`)
  } else {
    Wn_std <- NULL
  }
  
  list(W = W_std, b = b_std, W_next = Wn_std)
}

# 2) Permutation pro Dense-Layer
#    W: (input_dim × units), b: length units, W_next: (units × units_next)
canonical_permutation <- function(W, b, W_next = NULL) {
  # Score = L2-Norm der Spalte + Bias
  scores    <- sqrt(colSums(W^2) + b^2)
  order_idx <- order(scores)
  
  # Permutiere Spalten in W und Bias
  W_p  <- W[, order_idx, drop = FALSE]
  b_p  <- b[order_idx]
  
  # Und Zeilen in W_next
  if (!is.null(W_next)) {
    Wn_p <- W_next[order_idx, , drop = FALSE]
  } else {
    Wn_p <- NULL
  }
  
  list(W = W_p, b = b_p, W_next = Wn_p)
}

# Hauptfunktion: lädt ein SavedModel, wendet auf alle hidden layers
# zuerst sign-flip (inkl. Propagation), dann Permutation an.
# Speichert das Ergebnis mit "_canon.keras"-Suffix.
canonicalize_model <- function(model_path) {
  model    <- load_model_tf(model_path)
  weights  <- get_weights(model)
  n_layers <- length(weights) / 2
  
  # Extrahiere kernel/bias Paare
  kernels <- weights[seq(1, by = 2, length.out = n_layers)]
  biases  <- weights[seq(2, by = 2, length.out = n_layers)]
  
  # 1) Sign-flip auf hidden layers (i von 1 bis n_layers-1)
  for (i in seq_len(n_layers - 1)) {
    sf <- canonical_sign_flip(
      kernels[[i]],
      biases[[i]],
      kernels[[i + 1]]
    )
    kernels[[i]]     <- sf$W
    biases[[i]]      <- sf$b
    kernels[[i + 1]] <- sf$W_next
  }
  
  # 2) Permutation auf hidden layers
  for (i in seq_len(n_layers - 1)) {
    cp <- canonical_permutation(
      kernels[[i]],
      biases[[i]],
      kernels[[i + 1]]
    )
    kernels[[i]]     <- cp$W
    biases[[i]]      <- cp$b
    kernels[[i + 1]] <- cp$W_next
  }
  
  # Setze neue Gewichte und speichere Modell
  new_weights <- unlist(Map(function(K, B) list(K, B), kernels, biases), recursive = FALSE)
  set_weights(model, new_weights)
  
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
  
  if (is.null(output_file) == FALSE) {
    # Write representative paths to file
    writeLines(reps, con = output_file)
    message("Representatives written to: ", output_file)
  }
  
  # Return reps and centroid distances invisibly
  invisible(list(
    representatives     = reps,
    centroid_distances  = centroid_dist
  ))
}
