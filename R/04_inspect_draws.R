# 04_inspect_draws.R
# ──────────────────────────────────────────────────────────────────────────────
# Utilities for inspecting & plotting Bayesian posterior draws from DEI-MCMC

library(ggplot2)

# 1. Load draws ---------------------------------------------------------------
# path: the .rds file produced by run_dei_mcmc()
load_draws <- function(path) {
  draws <- readRDS(path)
  message("Loaded draws for parameters: ", paste(names(draws), collapse = ", "))
  return(draws)
}

# 2. Summarize a parameter ----------------------------------------------------
# draws: list returned by load_draws()
# param: character name of a scalar or vector param (e.g. "sigma" or "b4")
summarize_param <- function(draws, param) {
  if (!param %in% names(draws)) stop("Param not in draws.")
  vec <- as.vector(draws[[param]])
  cat("Summary of", param, ":\n")
  print(summary(vec))
  invisible(vec)
}

# 3. Traceplot of a scalar ----------------------------------------------------
traceplot_param <- function(draws, param) {
  vec <- summarize_param(draws, param)
  df  <- data.frame(iter = seq_along(vec), value = vec)
  ggplot(df, aes(x = iter, y = value)) +
    geom_line() +
    labs(title = paste("Traceplot:", param),
         x     = "Iteration",
         y     = param) +
    theme_minimal()
}

# 4. Density plot of a scalar -------------------------------------------------
density_param <- function(draws, param) {
  vec <- summarize_param(draws, param)
  df  <- data.frame(value = vec)
  ggplot(df, aes(x = value)) +
    geom_density(fill = "steelblue", alpha = 0.5) +
    labs(title = paste("Posterior density:", param),
         x     = param,
         y     = "Density") +
    theme_minimal()
}

# 5. Posterior‐predictive curves -----------------------------------------------
# Re‐implements your BNN forward pass for one draw index
predict_bnn <- function(x, draws, idx, H1, H2, H3) {
  # weights come in as matrices/vectors per draw
  W1 <- matrix(draws$W1[idx, ], nrow = H1)
  b1 <- draws$b1[idx, ]
  W2 <- matrix(draws$W2[idx, ], nrow = H2)
  b2 <- draws$b2[idx, ]
  W3 <- matrix(draws$W3[idx, ], nrow = H3)
  b3 <- draws$b3[idx, ]
  w4 <- draws$w4[idx, ]
  b4 <- draws$b4[idx]
  
  # a vectorized pass: x is vector length N
  N  <- length(x)
  a1 <- tanh(sweep(W1 %*% matrix(x, nrow = 1), 1, b1, `+`))
  a2 <- tanh(sweep(W2 %*% a1,           1, b2, `+`))
  a3 <- tanh(sweep(W3 %*% a2,           1, b3, `+`))
  mu <- as.vector(t(w4) %*% a3 + b4)
  return(mu)
}

# wrapper to plot many draws
plot_posterior_predictive <- function(draws,
                                      H1, H2, H3,
                                      x_grid = seq(-5, 5, length.out = 200),
                                      n_draws = 10) {
  total <- nrow(draws$W1)
  ids   <- sample(total, min(n_draws, total))
  df_list <- lapply(ids, function(i) {
    mu <- predict_bnn(x_grid, draws, i, H1, H2, H3)
    data.frame(x = x_grid, mu = mu, draw = factor(i))
  })
  df_all <- do.call(rbind, df_list)
  
  ggplot(df_all, aes(x = x, y = mu, group = draw)) +
    geom_line(alpha = 0.6) +
    labs(title = sprintf("Posterior predictive (%d draws)", length(unique(df_all$draw))),
         x     = "x",
         y     = "E[y|x]") +
    theme_minimal()
}


#Example
# 1) Load the draws
#getwd()
draws <- load_draws("results/mcmc_draws/sin_chains0_draws.rds")

# 2) Quick summaries
summarize_param(draws, "sigma")
traceplot_param(draws, "sigma")
density_param(draws,   "sigma")

# 3) Posterior predictive curves
plot_posterior_predictive(draws, H1=64, H2=64, H3=32, n_draws=5)
