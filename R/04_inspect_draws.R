# 04_inspect_draws.R
# ──────────────────────────────────────────────────────────────────────────────
# Utilities for inspecting & plotting Bayesian posterior draws from DEI-MCMC (cmdstanr)

library(ggplot2)
library(posterior)

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
  ggplot(df, aes(x=iter, y=value)) +
    geom_line() +
    labs(title = paste("Traceplot of", param),
         x = "Iteration", y = param) +
    theme_minimal()
}

# 4. Density plot of a scalar -------------------------------------------------
density_param <- function(draws_mat, param) {
  vec <- summarize_param(draws_mat, param)
  df  <- data.frame(value = vec)
  ggplot(df, aes(x=value)) +
    geom_density(fill = "steelblue", alpha = 0.5) +
    labs(title = paste("Posterior density of", param),
         x = param, y = "Density") +
    theme_minimal()
}

# 5. Posterior‐predictive curves -----------------------------------------------
#   Reconstructs network output for a few draws
plot_posterior_predictive <- function(draws_mat,
                                      H1, H2, H3,
                                      x_grid = seq(-5, 5, length.out=200),
                                      n_draws = 10) {
  # helper to extract parameter vector for draw i
  get_param_vec <- function(i, name_prefix, dims) {
    # dims: c(rows, cols) or length
    if (length(dims)==1) {
      cols <- paste0(name_prefix, "[", 1:dims, "]")
      return(as.numeric(draws_mat[i, cols]))
    } else {
      rows <- dims[1]; cols <- dims[2]
      names <- as.vector(outer(1:rows, 1:cols,
                               function(r,c) sprintf("%s[%d,%d]", name_prefix, r, c)))
      return(matrix(draws_mat[i, names], nrow=rows, byrow=FALSE))
    }
  }
  
  total <- nrow(draws_mat)
  ids   <- sample(total, min(n_draws, total))
  df_list <- lapply(ids, function(i) {
    W1 <- get_param_vec(i, "W1",  H1)
    b1 <- get_param_vec(i, "b1",  H1)
    W2 <- get_param_vec(i, "W2",  c(H2, H1))
    b2 <- get_param_vec(i, "b2",  H2)
    W3 <- get_param_vec(i, "W3",  c(H3, H2))
    b3 <- get_param_vec(i, "b3",  H3)
    w4 <- get_param_vec(i, "w4",  1)
    b4 <- get_param_vec(i, "b4",  1)
    
    # forward pass on x_grid
    a1 <- tanh(sweep(W1 %*% matrix(x_grid, nrow=1), 1, b1, `+`))
    a2 <- tanh(sweep(W2 %*% a1,           1,   b2, `+`))
    a3 <- tanh(sweep(W3 %*% a2,           1,   b3, `+`))
    mu <- as.numeric(w4 * a3 + b4)
    data.frame(x = x_grid, mu = mu, draw = factor(i))
  })
  df_all <- do.call(rbind, df_list)
  
  ggplot(df_all, aes(x=x, y=mu, group=draw)) +
    geom_line(alpha=0.6) +
    labs(title = sprintf("Posterior predictive (%d draws)", length(ids)),
         x = "x", y = "E[y|x]") +
    theme_minimal()
}

# Example usage (uncomment):
draws_mat <- load_draws("results/mcmc_draws/sin_1chains_draws.rds")
summarize_param(draws_mat, "sigma")
traceplot_param(draws_mat,   "sigma")
density_param(draws_mat,     "sigma")
plot_posterior_predictive(draws_mat, H1=64, H2=64, H3=32, n_draws=5)
