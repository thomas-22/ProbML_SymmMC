# 04_inspect_draws.R
# ──────────────────────────────────────────────────────────────────────────────
# Utilities for inspecting & plotting Bayesian posterior draws from DEI-MCMC (cmdstanr)

library(ggplot2)
library(posterior)
library(rlang)

# 1. Load draws ---------------------------------------------------------------
#   Reads a cmdstanr draws_array from .rds, converts to matrix
load_draws <- function(path) {
  draws_array <- readRDS(path)
  draws_mat   <- as_draws_matrix(draws_array)
  vars <- colnames(draws_mat)
  message("Loaded draws from: ", path)
  message("Available variables: ", paste(vars, collapse=", "))
  return(draws_mat)
}

# 2. Summarize a parameter ----------------------------------------------------
#   draws_mat: matrix from load_draws()
#   param: a single column name exactly as printed above
summarize_param <- function(draws_mat, param) {
  if (!param %in% colnames(draws_mat)) {
    stop("Parameter '", param, "' not found. Available: ",
         paste(colnames(draws_mat), collapse=", "))
  }
  vec <- draws_mat[, param]
  cat("Summary of", param, ":\n")
  print(summary(vec))
  invisible(vec)
}

# 3. Traceplot of a scalar ----------------------------------------------------
traceplot_param <- function(draws_mat, param) {
  vec <- summarize_param(draws_mat, param)
  df  <- data.frame(iter = seq_along(vec), value = vec)
  ggplot(df, aes(x = iter, y = .data[[param]])) +
    geom_line() +
    labs(
      title = paste("Traceplot of", param),
      x     = "Iteration",
      y     = param
    ) +
    theme_minimal()
}

# 4. Density plot of a scalar -------------------------------------------------
# draws_mat: a matrix of draws (rows = iterations, cols = vars)
# param:    a single variable name (e.g. "sigma")
density_param <- function(draws_mat, param) {
  if (!param %in% colnames(draws_mat)) {
    stop("Parameter '", param, "' not found. Available: ",
         paste(colnames(draws_mat), collapse = ", "))
  }
  vec <- draws_mat[, param, drop = TRUE]
  df  <- data.frame(value = vec)
  
  ggplot(df, aes(x = .data$value)) +
    geom_density(fill = "steelblue", alpha = 0.5) +
    labs(
      title = paste("Posterior density of", param),
      x     = param,
      y     = "Density"
    ) +
    theme_minimal()
}

# 5. Posterior–predictive mean curve (handles scalar vs indexed names) ------
plot_posterior_predictive_mean <- function(draws_mat,
                                           H1, H2, H3,
                                           x_grid  = seq(-5, 5, length.out = 200)) {
  total <- nrow(draws_mat)
  G     <- length(x_grid)
  mu_mat <- matrix(NA_real_, nrow = total, ncol = G)
  cn <- colnames(draws_mat)
  
  # helper to pull out either a scalar or length-d dims vector
  get_vec <- function(i, prefix, dims) {
    if (length(dims) == 1) {
      # first try bare name
      if (prefix %in% cn) {
        return(as.numeric(draws_mat[i, prefix]))
      }
      # otherwise try prefix[1] ... prefix[dims]
      cols <- paste0(prefix, "[", seq_len(dims), "]")
      if (!all(cols %in% cn)) {
        stop("Could not find columns for ", prefix)
      }
      return(as.numeric(draws_mat[i, cols]))
    } 
    # matrix case
    rows <- dims[1]; cols <- dims[2]
    names <- as.vector(outer(
      seq_len(rows), seq_len(cols),
      function(r,c) sprintf("%s[%d,%d]", prefix, r, c)
    ))
    if (!all(names %in% cn)) {
      stop("Could not find columns for ", prefix)
    }
    return(matrix(draws_mat[i, names], nrow = rows, byrow = FALSE))
  }
  
  for (i in seq_len(total)) {
    W1 <- get_vec(i, "W1",  H1); b1 <- get_vec(i, "b1",  H1)
    W2 <- get_vec(i, "W2", c(H2, H1)); b2 <- get_vec(i, "b2",  H2)
    W3 <- get_vec(i, "W3", c(H3, H2)); b3 <- get_vec(i, "b3",  H3)
    w4 <- get_vec(i, "w4", 1)            # will pick up "w4[1]" if needed
    b4 <- get_vec(i, "b4", 1)
    
    a1 <- tanh(sweep(W1 %*% matrix(x_grid,1), 1, b1, `+`))
    a2 <- tanh(sweep(W2 %*% a1,       1, b2, `+`))
    a3 <- tanh(sweep(W3 %*% a2,       1, b3, `+`))
    
    # collapse the H3 hidden units exactly as in Stan
    mu_mat[i, ] <- w4 * colSums(a3) + b4
  }
  
  mu_mean <- colMeans(mu_mat)
  df <- data.frame(x = x_grid, mu = mu_mean)
  
  ggplot(df, aes(x = x, y = mu)) +
    geom_line(color = "steelblue", size = 1) +
    labs(
      title = "Posterior predictive mean",
      x     = "x",
      y     = "E[y | x]"
    ) +
    theme_minimal()
}







# Example usage (uncomment):
#Works:
draws_mat <- load_draws("results/mcmc_draws/sin_1chains_draws.rds")
summarize_param(draws_mat, "sigma")
traceplot_param(draws_mat,   "sigma")
density_param(draws_mat,     "sigma")

plot_posterior_predictive_mean(draws_mat, H1=64, H2=64, H3=32)

