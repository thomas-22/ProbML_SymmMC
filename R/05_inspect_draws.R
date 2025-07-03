# 05_inspect_draws.R
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

# 2. Summarize a parameter (works on draws_df, draws_array, or draws_matrix)  
summarize_param <- function(draws, param) {
  # convert file paths → matrix if needed
  mat <- if (is.character(draws)) {
    load_and_combine(draws)
  } else if (inherits(draws, c("draws_array","draws_df","draws_matrix"))) {
    as_draws_matrix(draws)
  } else if (is.matrix(draws)) {
    draws
  } else {
    stop("Unsupported draws object; supply a matrix, draws_* or file vector")
  }
  if (!param %in% colnames(mat)) {
    stop("Parameter '",param,"' not found. Available: ",
         paste(colnames(mat), collapse=", "))
  }
  vec <- mat[, param]
  cat("Summary of", param, ":\n")
  print(summary(vec))
  invisible(vec)
}

# 3. Traceplot of a scalar (uses summarize_param under the hood)  
traceplot_param <- function(draws, param) {
  vec <- summarize_param(draws, param)
  df  <- data.frame(iter = seq_along(vec), value = vec)
  ggplot(df, aes(x = iter, y = value)) +
    geom_line() +
    labs(title = paste("Traceplot of", param),
         x = "Iteration", y = param) +
    theme_minimal()
}

# 4. Density plot of a scalar -------------------------------------------------
# draws_mat: a matrix of draws (rows = iterations, cols = vars)
# param:    a single variable name (e.g. "sigma")
density_param <- function(draws, param) {
  mat <- if (is.character(draws)) {
    load_and_combine(draws)
  } else if (inherits(draws, c("draws_array","draws_df","draws_matrix"))) {
    as_draws_matrix(draws)
  } else if (is.matrix(draws)) {
    draws
  } else stop("Unsupported draws object")
  if (!param %in% colnames(mat)) stop("Param not found")
  df <- data.frame(value = mat[, param])
  ggplot(df, aes(x = value)) +
    geom_density(fill = "steelblue", alpha = 0.5) +
    labs(title = paste("Posterior density of", param),
         x = param, y = "Density") +
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

plot_pp_mean_facets <- function(draw_files, H1, H2, H3,
                                x_grid = seq(-5, 5, length.out = 200)) {
  # helper to pull out scalars, vectors or matrices from a flat draws matrix
  get_vec <- function(mat, i, prefix, dims) {
    cn <- colnames(mat)
    if (length(dims) == 1) {
      # scalar or length-dims vector
      if (prefix %in% cn) return(mat[i, prefix])
      cols <- paste0(prefix, "[", seq_len(dims), "]")
      return(as.numeric(mat[i, cols]))
    }
    # matrix case
    rows <- dims[1]; cols <- dims[2]
    names <- as.vector(outer(
      seq_len(rows), seq_len(cols),
      function(r, c) sprintf("%s[%d,%d]", prefix, r, c)
    ))
    matrix(mat[i, names], nrow = rows, byrow = FALSE)
  }
  
  # process each chain separately
  chain_dfs <- map2(draw_files, seq_along(draw_files), function(path, cid) {
    # load draws and turn into a plain matrix
    draws_df <- as_draws_df(load_draws(path))
    mat      <- as_draws_matrix(as_draws_array(draws_df))
    mat      <- as.matrix(mat)
    n_draws  <- nrow(mat)
    G        <- length(x_grid)
    
    # compute mu for every draw × grid point
    mu_list <- vector("list", n_draws)
    for (i in seq_len(n_draws)) {
      W1 <- get_vec(mat, i, "W1",  H1);   b1 <- get_vec(mat, i, "b1",  H1)
      W2 <- get_vec(mat, i, "W2",  c(H2, H1)); b2 <- get_vec(mat, i, "b2", H2)
      W3 <- get_vec(mat, i, "W3",  c(H3, H2)); b3 <- get_vec(mat, i, "b3", H3)
      w4 <- get_vec(mat, i, "w4",  H3);       b4 <- get_vec(mat, i, "b4", 1)
      
      A1 <- tanh( outer(W1, x_grid, `*`) + b1 )
      A2 <- tanh( W2 %*% A1 + b2 )
      A3 <- tanh( W3 %*% A2 + b3 )
      
      # drop() ensures we get a plain numeric vector length G
      mu_list[[i]] <- drop(w4 %*% A3) + b4
    }
    mu_mat <- do.call(rbind, mu_list)
    
    # turn into long form and compute chain‐wise mean
    tibble(
      chain = paste0("chain", cid),
      draw  = rep(seq_len(n_draws), each = G),
      x     = rep(x_grid, times = n_draws),
      mu    = as.vector(t(mu_mat))
    ) %>%
      group_by(chain, x) %>%
      mutate(mu_mean = mean(mu)) %>%
      ungroup()
  })
  
  df_all <- bind_rows(chain_dfs)
  
  ggplot(df_all, aes(x = x)) +
    # all posterior‐predictive samples
    geom_line(aes(y = mu, group = interaction(chain, draw)),
              color = "blue", size = 0.2, alpha = 0.1) +
    # bold mean per chain
    geom_line(aes(y = mu_mean), color = "red", size = 1) +
    facet_wrap(~ chain, ncol = 2) +
    labs(
      title = "Posterior Predictive MCMC Samples and Mean by Chain, sinusoidal",
      x     = "x",
      y     = "E[y|x]"
    ) +
    theme_minimal()
}

plot_posterior_predictive_samples <- function(
    draws_mat,
    H1, H2, H3,
    num_draws = 20,
    x_grid    = seq(-5, 5, length.out = 200),
    seed      = 42
) {
  library(ggplot2)
  library(posterior)
  
  # helper to extract parameters from draws_mat
  get_vec <- function(i, prefix, dims) {
    cn <- colnames(draws_mat)
    if (length(dims) == 1 && dims == 1) {
      # scalar
      if (prefix %in% cn) return(as.numeric(draws_mat[i, prefix]))
      cols <- paste0(prefix, "[", seq_len(1), "]")
      return(as.numeric(draws_mat[i, cols]))
    } else if (length(dims) == 1) {
      # vector length dims
      cols <- paste0(prefix, "[", seq_len(dims), "]")
      return(as.numeric(draws_mat[i, cols]))
    } else {
      # matrix dims = c(rows, cols)
      rows <- dims[1]; cols <- dims[2]
      names <- as.vector(outer(
        seq_len(rows), seq_len(cols),
        function(r,c) sprintf("%s[%d,%d]", prefix, r, c)
      ))
      return(matrix(draws_mat[i, names], nrow = rows, byrow = FALSE))
    }
  }
  
  total <- nrow(draws_mat)
  set.seed(seed)
  idxs  <- sample(seq_len(total), min(num_draws, total))
  
  # build sample curves
  pp_samples <- do.call(rbind, lapply(idxs, function(i) {
    W1 <- get_vec(i, "W1", H1); b1 <- get_vec(i, "b1", H1)
    W2 <- get_vec(i, "W2", c(H2, H1)); b2 <- get_vec(i, "b2", H2)
    W3 <- get_vec(i, "W3", c(H3, H2)); b3 <- get_vec(i, "b3", H3)
    w4 <- get_vec(i, "w4", H3);             b4 <- get_vec(i, "b4", 1)
    a1 <- tanh(sweep(W1 %*% matrix(x_grid,1), 1, b1, `+`))
    a2 <- tanh(sweep(W2 %*% a1,       1, b2, `+`))
    a3 <- tanh(sweep(W3 %*% a2,       1, b3, `+`))
    mu <- as.numeric(w4 %*% a3 + b4)
    data.frame(draw = i, x = x_grid, mu = mu)
  }))
  
  # compute predictive mean
  mu_mat <- sapply(seq_len(total), function(i) {
    W1 <- get_vec(i, "W1", H1); b1 <- get_vec(i, "b1", H1)
    W2 <- get_vec(i, "W2", c(H2, H1)); b2 <- get_vec(i, "b2", H2)
    W3 <- get_vec(i, "W3", c(H3, H2)); b3 <- get_vec(i, "b3", H3)
    w4 <- get_vec(i, "w4", H3);             b4 <- get_vec(i, "b4", 1)
    a1 <- tanh(sweep(W1 %*% matrix(x_grid,1), 1, b1, `+`))
    a2 <- tanh(sweep(W2 %*% a1,       1, b2, `+`))
    a3 <- tanh(sweep(W3 %*% a2,       1, b3, `+`))
    as.numeric(w4 %*% a3 + b4)
  })
  mu_mean <- rowMeans(mu_mat)
  mean_df <- data.frame(x = x_grid, mu = mu_mean)
  
  # final plot
  ggplot() +
    geom_line(
      data = pp_samples,
      aes(x = x, y = mu, group = draw),
      alpha = 0.3
    ) +
    geom_line(
      data = mean_df,
      aes(x = x, y = mu),
      color = "steelblue", size = 1
    ) +
    labs(
      title = sprintf(
        "Posterior Predictive Samples (n=%d) + Mean", num_draws
      ),
      x = "x", y = "E[y | x]"
    ) +
    theme_minimal()
}

show_sin_full_analysis <- function(
    dataset_name,
    data_dir     = "data/synthetic",
    model_paths,
    draw_file,
    num_draws = 20,
    H1 = 64, H2 = 64, H3 = 32
) {
  library(ggplot2)
  library(dplyr)
  
  # 1) Load dataset and true function
  data_file <- file.path(data_dir, paste0(dataset_name, "_dataset.rds"))
  df        <- readRDS(data_file)
  x_grid    <- seq(min(df$x), max(df$x), length.out = 200)
  true_df   <- tibble(x = x_grid, y_true = sin(2 * pi * x_grid / 2))
  
  # 2) Compute ensemble predictions
  ensemble_df <- purrr::map_dfr(model_paths, function(mp) {
    model <- load_model_tf(mp)
    preds <- as.numeric(model %>% predict(matrix(x_grid, ncol = 1)))
    tibble(x = x_grid, y_pred = preds, model = basename(mp))
  })
  
  # 3) Compute posterior sample curves + mean
  draws_mat <- load_draws(draw_file)
  # helper to extract parameters
  get_vec <- function(i, prefix, dims) {
    cn <- colnames(draws_mat)
    if (length(dims)==1 && dims>1) {
      cols <- paste0(prefix, "[", seq_len(dims), "]")
      return(as.numeric(draws_mat[i, cols]))
    } else if (length(dims)==1) {
      name <- if (prefix %in% cn) prefix else paste0(prefix, "[1]")
      return(as.numeric(draws_mat[i, name]))
    }
    rows <- dims[1]; cols <- dims[2]
    names <- as.vector(outer(
      seq_len(rows), seq_len(cols),
      function(r,c) sprintf("%s[%d,%d]", prefix, r, c)
    ))
    matrix(draws_mat[i, names], nrow = rows)
  }
  total <- nrow(draws_mat)
  set.seed(123)
  ids   <- sample(total, min(num_draws, total))
  
  pp_samples <- purrr::map_dfr(ids, function(i) {
    W1 <- get_vec(i, "W1", H1); b1 <- get_vec(i, "b1", H1)
    W2 <- get_vec(i, "W2", c(H2, H1)); b2 <- get_vec(i, "b2", H2)
    W3 <- get_vec(i, "W3", c(H3, H2)); b3 <- get_vec(i, "b3", H3)
    w4 <- get_vec(i, "w4", H3);             b4 <- get_vec(i, "b4", 1)
    a1 <- tanh(sweep(W1 %*% matrix(x_grid,1), 1, b1, `+`))
    a2 <- tanh(sweep(W2 %*% a1,       1, b2, `+`))
    a3 <- tanh(sweep(W3 %*% a2,       1, b3, `+`))
    mu <- as.numeric(w4 %*% a3 + b4)
    tibble(draw = i, x = x_grid, y_pp = mu)
  })
  mu_mat <- purrr::map_dfc(seq_len(total), function(i) {
    W1 <- get_vec(i, "W1", H1); b1 <- get_vec(i, "b1", H1)
    W2 <- get_vec(i, "W2", c(H2, H1)); b2 <- get_vec(i, "b2", H2)
    W3 <- get_vec(i, "W3", c(H3, H2)); b3 <- get_vec(i, "b3", H3)
    w4 <- get_vec(i, "w4", H3);             b4 <- get_vec(i, "b4", 1)
    a1 <- tanh(sweep(W1 %*% matrix(x_grid,1), 1, b1, `+`))
    a2 <- tanh(sweep(W2 %*% a1,       1, b2, `+`))
    a3 <- tanh(sweep(W3 %*% a2,       1, b3, `+`))
    as.numeric(w4 %*% a3 + b4)
  })
  mean_df <- tibble(x = x_grid, y_mean = rowMeans(mu_mat))
  
  # 4) Plot everything with adjusted sizes
  ggplot() +
    # data points smaller
    geom_point(
      data = df, aes(x = x, y = y),
      size = 0.7, alpha = 0.4, color = "blue"
    ) +
    # true process dashed
    geom_line(
      data = true_df, aes(x = x, y = y_true),
      linetype = "dashed", color = "green"
    ) +
    # ensemble preds thinner
    geom_line(
      data = ensemble_df, aes(x = x, y = y_pred, color = model),
      size = 0.5
    ) +
    # posterior samples
    geom_line(
      data = pp_samples, aes(x = x, y = y_pp, group = draw),
      alpha = 0.3
    ) +
    # posterior mean
    geom_line(
      data = mean_df, aes(x = x, y = y_mean),
      color = "black", size = 0.7
    ) +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = paste0("Full Analysis for '", dataset_name, "'"),
      x = "x", y = "y"
    ) +
    theme_minimal()
}

traceplot_all_chains <- function(da, pars = NULL, ncol = 2) {
  # convert to tibble
  df <- as_draws_df(da)
  
  # optionally keep only a subset of parameters
  if (!is.null(pars)) {
    keep_cols <- c(".chain", ".iteration", pars)
    df <- df %>% select(any_of(keep_cols))
  }
  
  # pivot to long format
  df_long <- df %>%
    pivot_longer(
      cols = -c(.chain, .iteration),
      names_to  = "parameter",
      values_to = "value"
    )
  
  # ggplot traceplot
  ggplot(df_long, aes(x = .iteration, y = value, color = factor(.chain))) +
    geom_line(alpha = 0.7) +
    facet_wrap(~ parameter, scales = "free_y", ncol = ncol) +
    labs(
      x     = "Iteration",
      color = "Chain"
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(face = "bold"),
      legend.position = "bottom"
    )
}


# ---------------------------------------------------------------------------
# Credible-band plot for the synthetic (sinusoidal)
# ---------------------------------------------------------------------------
#  * draw_files …… character vector with one *.rds file per MCMC chain
#  * H1,H2,H3 …… hidden-layer sizes used in Stan
#  * noise_sd  …… known σ of the additive Gaussian noise (0.2 in my study)
#  * level     …… credible level, default 0.90  (= two–sided 90 % interval)
#  * x_grid    …… evaluation grid for the curve & ribbons
# ---------------------------------------------------------------------------
plot_credible_band_synth <- function(draw_files,
                                     nn_paths,
                                     dataset_rds_path,
                                     H1, H2, H3,
                                     noise_sd = 0.2,
                                     level    = 0.90,
                                     x_grid   = seq(-5, 5, length.out = 300))
{
  library(posterior)
  library(dplyr)
  library(ggplot2)
  library(keras)
  
  # ─────────────────────────────────────────────────────────────────────────────
  # 1. Load observed data
  df_obs <- readRDS(dataset_rds_path)
  # expect df_obs to have columns x and y
  
  # ─────────────────────────────────────────────────────────────────────────────
  # 2. Stack the MCMC chains
  dfs <- lapply(seq_along(draw_files), function(j){
    as_draws_df(load_draws(draw_files[j])) %>%
      mutate(.chain = j, .iteration = row_number())
  })
  mat <- as_draws_matrix(bind_rows(dfs))
  cn  <- colnames(mat)
  getp <- function(i, prefix, dims) {
    # scalar or vector
    if (length(dims) == 1) {
      if (prefix %in% cn) {
        return(as.numeric(mat[i, prefix]))
      }
      cols <- sprintf("%s[%d]", prefix, seq_len(dims))
      return(as.numeric(mat[i, cols]))
    }
    # matrix case
    rows <- dims[1]; cols <- dims[2]
    names <- as.vector(outer(
      seq_len(rows), seq_len(cols),
      function(r, c) sprintf("%s[%d,%d]", prefix, r, c)
    ))
    return(matrix(mat[i, names], nrow = rows, byrow = FALSE))
  }
  
  # ─────────────────────────────────────────────────────────────────────────────
  # 3. Build posterior band for f(x)
  preds <- sapply(x_grid, \(x0){
    xv<-matrix(x0,1)
    sapply(1:nrow(mat), \(i){
      W1<-getp(i,"W1",H1); b1<-getp(i,"b1",H1)
      W2<-getp(i,"W2",c(H2,H1)); b2<-getp(i,"b2",H2)
      W3<-getp(i,"W3",c(H3,H2)); b3<-getp(i,"b3",H3)
      w4<-getp(i,"w4",H3);       b4<-getp(i,"b4",1)
      a1<-tanh(W1%*%xv+b1); a2<-tanh(W2%*%a1+b2); a3<-tanh(W3%*%a2+b3)
      as.numeric(w4%*%a3+b4)
    })
  })
  qlo <- apply(preds, 2, quantile, (1 - level)/2)
  qhi <- apply(preds, 2, quantile, 1 - (1 - level)/2)
  qmd <- apply(preds, 2, median)
  band <- tibble(x = x_grid, lo = qlo, mid = qmd, hi = qhi)
  
  # ─────────────────────────────────────────────────────────────────────────────
  # 4. Ensemble-mean curve in red
  ens <- sapply(x_grid, function(xx) {
    mean(sapply(nn_paths, function(p) {
      as.numeric(load_model_tf(p) %>% predict(matrix(xx, ncol = 1)))
    }))
  })
  ens_df <- tibble(x = x_grid, y = ens)
  
  # ─────────────────────────────────────────────────────────────────────────────
  # 5. True process + noise envelope (green dashed)
  tru <- tibble(
    x    = x_grid,
    mean = sin(pi * x_grid),
    lo   = sin(pi * x_grid) - qnorm(1 - (1 - level)/2) * noise_sd,
    hi   = sin(pi * x_grid) + qnorm(1 - (1 - level)/2) * noise_sd
  )
  
  # ─────────────────────────────────────────────────────────────────────────────
  # 6. Plot all layers
  ggplot() +
    # observed points in blue
    geom_point(data = df_obs,
               aes(x = x, y = y),
               color = "blue", alpha = 0.25, size = 0.7) +
    # ensemble mean in red
    geom_line(data = ens_df,
              aes(x = x, y = y, color = "Ensemble mean"),
              linewidth = 1.1) +
    # posterior credible ribbon
    geom_ribbon(data = band,
                aes(x = x, ymin = lo, ymax = hi, fill = "Posterior 90%"),
                alpha = 0.25, colour = NA) +
    # posterior mean thicker
    geom_line(data = band,
              aes(x = x, y = mid, color = "Posterior mean"),
              linewidth = 1.2, alpha = 0.8) +
    # true noise envelope
    geom_line(data = tru,
              aes(x = x, y = lo, color = "True 90% noise"),
              linewidth = 0.7, linetype = "dashed") +
    geom_line(data = tru,
              aes(x = x, y = hi, color = "True 90% noise"),
              linewidth = 0.7, linetype = "dashed") +
    # true function
    geom_line(data = tru,
              aes(x = x, y = mean, color = "True function"),
              linetype = "dashed", linewidth = 0.9) +
    # scales & legend
    scale_fill_manual("", values = c("Posterior 90%" = "darkblue")) +
    scale_color_manual("", values = c(
      "Posterior mean"  = "darkblue",
      "True function"   = "red",
      "True 90% noise"  = "red",
      "Ensemble mean"   = "orange"
    )) +
    labs(
      title = sprintf("%d%% Credible Band vs. Sinusoid (σ=%.2f), 4 chains", level*100, noise_sd),
      x     = "x",
      y     = "y"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
}






# ──────────────────────────────────────────────────────────────────────────────
# Utilities for UCI-Airfoil DEI-MCMC posterior inspection

library(ggplot2)
library(dplyr)
library(tidyr)
library(posterior)

# Helper: extract parameter from draws matrix --------------------------------
# draws_mat: posterior::draws_matrix (rows = iterations, cols = vars)
get_param <- function(draws_mat, iter, prefix, dims) {
  cn <- colnames(draws_mat)
  if (length(dims) == 1) {
    # scalar or vector
    if (dims == 1) {
      # scalar: try prefix or prefix[1]
      nm <- if (prefix %in% cn) prefix else paste0(prefix, "[1]")
      return(as.numeric(draws_mat[iter, nm]))
    }
    # vector of length dims
    cols <- paste0(prefix, "[", seq_len(dims), "]")
    return(as.numeric(draws_mat[iter, cols]))
  }
  # matrix dims = c(rows, cols)
  rows <- dims[1]; cols <- dims[2]
  names <- as.vector(outer(
    seq_len(rows), seq_len(cols),
    function(r,c) sprintf("%s[%d,%d]", prefix, r, c)
  ))
  mat <- matrix(draws_mat[iter, names], nrow = rows, byrow = FALSE)
  return(mat)
}

# 1) Partial Dependence: Posterior Predictive Mean ----------------------------
plot_posterior_predictive_mean_uciairfoil <- function(
    draw_file,
    dataset_rds_path = "data/uci/airfoil_dataset.rds",
    scaler_rds_path  = "data/uci/airfoil_scaler.rds",
    H1, H2, H3,
    n.grid = 100
) {
  # load posterior draws
  draws_mat <- load_draws(draw_file)
  
  # load raw + scaler
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  # compute feature means in raw/log space
  raw_means <- df_raw %>%
    mutate(Frequency = log1p(Frequency)) %>%
    summarise(across(all_of(features), mean, na.rm=TRUE))
  
  # build a grid + compute posterior mean for each feature
  pd_list <- purrr::map_df(features, function(feat) {
    # raw grid for this feature
    x.seq <- seq(min(df_raw[[feat]], na.rm=TRUE),
                 max(df_raw[[feat]], na.rm=TRUE),
                 length.out = n.grid)
    
    # replicate mean row
    dfg <- raw_means[rep(1, n.grid), ] %>% as.data.frame()
    dfg[[feat]] <- x.seq
    
    # preprocess: log + scale
    dfg$Frequency <- log1p(dfg$Frequency)
    X_raw <- as.matrix(dfg[, features])
    X_scaled <- sweep(X_raw,  2, scaler$feature_means, "-")
    X_scaled <- sweep(X_scaled, 2, scaler$feature_sds,   "/")
    
    # allocate container
    n_iter <- nrow(draws_mat)
    mu_mat <- matrix(NA_real_, nrow = n_iter, ncol = n.grid)
    
    # loop over iterations
    for (i in seq_len(n_iter)) {
      W1 <- get_param(draws_mat, i, "W1", c(H1, length(features)))
      b1 <- get_param(draws_mat, i, "b1", H1)
      W2 <- get_param(draws_mat, i, "W2", c(H2, H1))
      b2 <- get_param(draws_mat, i, "b2", H2)
      W3 <- get_param(draws_mat, i, "W3", c(H3, H2))
      b3 <- get_param(draws_mat, i, "b3", H3)
      w4 <- get_param(draws_mat, i, "w4", H3)
      b4 <- get_param(draws_mat, i, "b4", 1)
      
      # forward pass for all grid points at once
      A1 <- tanh(W1 %*% t(X_scaled) + b1)
      A2 <- tanh(W2 %*% A1 + b2)
      A3 <- tanh(W3 %*% A2 + b3)
      # w4 is length H3: broadcast
      mu_mat[i, ] <- as.numeric(crossprod(w4, A3) + b4)
    }
    
    # posterior predictive mean (on scaled target)
    mu_mean_scaled <- colMeans(mu_mat)
    # un-scale to original SoundPressure
    mu_mean <- mu_mean_scaled * scaler$target_sd + scaler$target_mean
    
    tibble(
      Feature   = feat,
      Value     = x.seq,
      Predicted = mu_mean
    )
  })
  
  # raw scatter
  scatter_df <- df_raw %>%
    rename(Actual = SoundPressure) %>%
    mutate(Frequency = Frequency) %>%
    pivot_longer(all_of(features),
                 names_to  = "Feature",
                 values_to = "Value")
  
  ggplot() +
    geom_point(data = scatter_df,
               aes(x = Value, y = Actual),
               size = 0.6, alpha = 0.4, color = "blue") +
    geom_line(data = pd_list,
              aes(x = Value, y = Predicted),
              color = "red", size = 1) +
    facet_wrap(~ Feature, scales = "free_x") +
    labs(
      title = "Airfoil DEI-MCMC: Partial Dependence (Posterior Mean)",
      x     = "Feature value",
      y     = "SoundPressure"
    ) +
    theme_minimal()
}

# 2) Partial Dependence: Posterior Predictive Samples ------------------------
plot_posterior_predictive_samples_uciairfoil <- function(
    draw_file,
    dataset_rds_path = "data/uci/airfoil_dataset.rds",
    scaler_rds_path  = "data/uci/airfoil_scaler.rds",
    H1, H2, H3,
    n.grid     = 100,
    num_draws  = 20,
    seed       = 42
) {
  set.seed(seed)
  draws_mat <- load_draws(draw_file)
  total     <- nrow(draws_mat)
  ids       <- sample(seq_len(total), min(num_draws, total))
  
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  raw_means <- df_raw %>%
    mutate(Frequency = log1p(Frequency)) %>%
    summarise(across(all_of(features), mean, na.rm=TRUE))
  
  # build sample curves
  samples_df <- purrr::map_dfr(ids, function(i) {
    purrr::map_df(features, function(feat) {
      x.seq <- seq(min(df_raw[[feat]], na.rm=TRUE),
                   max(df_raw[[feat]], na.rm=TRUE),
                   length.out = n.grid)
      dfg <- raw_means[rep(1, n.grid), ] %>% as.data.frame()
      dfg[[feat]] <- x.seq
      dfg$Frequency <- log1p(dfg$Frequency)
      X_raw <- as.matrix(dfg[, features])
      X_scaled <- sweep(X_raw, 2, scaler$feature_means, "-")
      X_scaled <- sweep(X_scaled, 2, scaler$feature_sds,   "/")
      
      # extract params for this draw
      W1 <- get_param(draws_mat, i, "W1", c(H1, length(features)))
      b1 <- get_param(draws_mat, i, "b1", H1)
      W2 <- get_param(draws_mat, i, "W2", c(H2, H1))
      b2 <- get_param(draws_mat, i, "b2", H2)
      W3 <- get_param(draws_mat, i, "W3", c(H3, H2))
      b3 <- get_param(draws_mat, i, "b3", H3)
      w4 <- get_param(draws_mat, i, "w4", H3)
      b4 <- get_param(draws_mat, i, "b4", 1)
      
      A1 <- tanh(W1 %*% t(X_scaled) + b1)
      A2 <- tanh(W2 %*% A1 + b2)
      A3 <- tanh(W3 %*% A2 + b3)
      mu_scaled <- as.numeric(crossprod(w4, A3) + b4)
      mu <- mu_scaled * scaler$target_sd + scaler$target_mean
      
      tibble(
        draw      = i,
        Feature   = feat,
        Value     = x.seq,
        Predicted = mu
      )
    })
  })
  
  # raw scatter
  scatter_df <- df_raw %>%
    rename(Actual = SoundPressure) %>%
    pivot_longer(all_of(features),
                 names_to  = "Feature",
                 values_to = "Value")
  
  ggplot() +
    geom_point(data = scatter_df,
               aes(x = Value, y = Actual),
               size = 0.6, alpha = 0.4, color = "blue") +
    geom_line(data = samples_df,
              aes(x = Value, y = Predicted, group = draw),
              alpha = 0.1, color = "blue", size = 0.2) +
    facet_wrap(~ Feature, scales = "free_x") +
    labs(
      title = "Airfoil DEI-MCMC: Partial Dependence (Posterior Samples)",
      x     = "Feature value",
      y     = "SoundPressure"
    ) +
    theme_minimal()
}

plot_posterior_and_single_dei <- function(
    draw_file,
    nn_model_path,
    dataset_rds_path  = "data/uci/airfoil_dataset.rds",
    scaler_rds_path   = "data/uci/airfoil_scaler.rds",
    H1, H2, H3,
    n.grid     = 100,
    num_draws  = 20,
    nn_psample_color = "red"
) {
  # ---------------------------------------------------------------------------
  # 1) base posterior-predictive plot (no colour scales yet)
  # ---------------------------------------------------------------------------
  p_base <- plot_posterior_predictive_samples_uciairfoil_color(
    draw_file         = draw_file,
    dataset_rds_path  = dataset_rds_path,
    scaler_rds_path   = scaler_rds_path,
    H1 = H1, H2 = H2, H3 = H3,
    n.grid    = n.grid,
    num_draws = num_draws,
    sample_color = nn_psample_color
  )
  
  # ---------------------------------------------------------------------------
  # 2) load raw data + scaler (for partial-dependence of the single net)
  # ---------------------------------------------------------------------------
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  raw_means <- df_raw %>%
    dplyr::mutate(Frequency = log1p(Frequency)) %>%
    dplyr::summarise(dplyr::across(dplyr::all_of(features), mean, na.rm = TRUE))
  
  # ---------------------------------------------------------------------------
  # 3) load the chosen Keras model
  # ---------------------------------------------------------------------------
  model <- load_model_tf(nn_model_path)
  
  # ---------------------------------------------------------------------------
  # 4) partial-dependence predictions for that single network
  # ---------------------------------------------------------------------------
  nn_df <- purrr::map_dfr(features, function(feat) {
    x.seq <- seq(min(df_raw[[feat]], na.rm = TRUE),
                 max(df_raw[[feat]], na.rm = TRUE),
                 length.out = n.grid)
    
    newdata <- raw_means[rep(1, n.grid), ] %>% as.data.frame()
    newdata[[feat]] <- x.seq
    newdata$Frequency <- log1p(newdata$Frequency)
    
    X_raw    <- as.matrix(newdata[, features])
    X_scaled <- sweep(X_raw, 2, scaler$feature_means, "-")
    X_scaled <- sweep(X_scaled, 2, scaler$feature_sds,   "/")
    
    preds <- as.numeric(model %>% predict(X_scaled))
    preds <- preds * scaler$target_sd + scaler$target_mean
    
    tibble::tibble(
      Feature   = feat,
      Value     = x.seq,
      Predicted = preds
    )
  })
  
  # ---------------------------------------------------------------------------
  # 5) overlay net + single colour scale & legend
  # ---------------------------------------------------------------------------
  new_color <- darken_color(nn_psample_color, amount = 0.30)
  
  p_base +
    geom_line(
      data = nn_df,
      aes(x = Value, y = Predicted, colour = "Neural net"),
      size = 1.1
    ) +
    ## single, explicit colour scale (fixed order!)
    scale_colour_manual(
      values = c(
        "Actual data"           = "blue",
        "Neural net"            = new_color,
        "Posterior mean"        = nn_psample_color,
        "90% CI upper"          = nn_psample_color,
        "90% CI lower"          = nn_psample_color,
        "Posterior samples"     = nn_psample_color
      ),
      breaks = c("Actual data",
                 "Neural net",
                 "Posterior mean",
                 "90% CI upper",
                 "90% CI lower",
                 "Posterior samples"),
      name   = NULL
    ) +
    guides(
      colour = guide_legend(
        override.aes = list(
          linetype = c("blank", "solid", "solid", "dashed", "dashed", "solid"),
          shape    = c(16, NA, NA, NA, NA, NA),
          size     = c(2, 1.1, 1, 0.6, 0.6, 0.2),
          alpha    = c(0.5, 1, 1, 1, 1, 0.1)
        )
      )
    ) +
    labs(
      title = paste0("Airfoil PD & DEI (model = ", basename(nn_model_path), ")"),
      x     = "Feature value",
      y     = "SoundPressure"
    ) +
    theme(
      legend.position = "bottom",
      legend.title    = element_blank()
    )
}


# Reads a text file of keras model paths, overlays their PD curves on the posterior-sample plot
plot_posterior_ensemble_and_dei <- function(
  draw_file,
  nn_model_path,
  dataset_rds_path  = "data/uci/airfoil_dataset.rds",
  scaler_rds_path   = "data/uci/airfoil_scaler.rds",
  H1, H2, H3,
  n.grid     = 100,
  num_draws  = 20,
  nn_psample_color = "red"
) {
  # 1) base posterior‐predictive plot, force it into our colour
  p_base <- plot_posterior_predictive_samples_uciairfoil(
    draw_file         = draw_file,
    dataset_rds_path  = dataset_rds_path,
    scaler_rds_path   = scaler_rds_path,
    H1 = H1, H2 = H2, H3 = H3,
    n.grid    = n.grid,
    num_draws = num_draws
  ) +
    # override any default sample colour
    scale_colour_identity() +
    scale_fill_identity()
  
  # 2) load raw data & scaler
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  raw_means <- df_raw %>%
    mutate(Frequency = log1p(Frequency)) %>%
    summarise(across(all_of(features), mean, na.rm = TRUE))
  
  # 3) load *one* Keras model
  model <- load_model_tf(nn_model_path)
  
  # 4) build its partial‐dependence predictions
  nn_df <- map_dfr(features, function(feat) {
    x.seq <- seq(
      min(df_raw[[feat]], na.rm=TRUE),
      max(df_raw[[feat]], na.rm=TRUE),
      length.out = n.grid
    )
    newdata <- raw_means[rep(1, n.grid), ] %>% as.data.frame()
    newdata[[feat]] <- x.seq
    newdata$Frequency <- log1p(newdata$Frequency)
    
    X_raw    <- as.matrix(newdata[, features])
    X_scaled <- sweep(X_raw, 2, scaler$feature_means, "-")
    X_scaled <- sweep(X_scaled, 2, scaler$feature_sds,   "/")
    
    preds <- as.numeric(model %>% predict(X_scaled))
    preds <- preds * scaler$target_sd + scaler$target_mean
    
    tibble(
      Feature   = feat,
      Value     = x.seq,
      Predicted = preds
    )
  })
  
  # 5) overlay the single‐net PD line, in the same colour
  p_base +
    geom_line(
      data = nn_df,
      aes(x = Value, y = Predicted),
      colour = nn_psample_color,
      size   = 1.1
    ) +
    labs(
      title = paste0("Airfoil PD & DEI (model = ", basename(nn_model_path), ")"),
      x     = "Feature value",
      y     = "SoundPressure"
    ) +
    theme(legend.position = "none")
}

plot_posterior_predictive_samples_uciairfoil_color <- function(
    draw_file,
    dataset_rds_path = "data/uci/airfoil_dataset.rds",
    scaler_rds_path  = "data/uci/airfoil_scaler.rds",
    H1, H2, H3,
    n.grid     = 100,
    num_draws  = 20,
    seed       = 42,
    sample_color = "blue"     # kept for compatibility (set via wrapper)
) {
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(purrr)
  
  set.seed(seed)
  draws_mat <- load_draws(draw_file)
  ids       <- sample(seq_len(nrow(draws_mat)), min(num_draws, nrow(draws_mat)))
  
  df_raw   <- readRDS(dataset_rds_path)
  scaler   <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  raw_means <- df_raw %>%
    mutate(Frequency = log1p(Frequency)) %>%
    summarise(across(all_of(features), mean, na.rm = TRUE))
  
  # ---------------- posterior-sample curves -------------------------
  samples_df <- purrr::map_dfr(ids, function(i) {
    purrr::map_dfr(features, function(feat) {
      x.seq <- seq(min(df_raw[[feat]], na.rm = TRUE),
                   max(df_raw[[feat]], na.rm = TRUE),
                   length.out = n.grid)
      dfg <- raw_means[rep(1, n.grid), ] %>% as.data.frame()
      dfg[[feat]] <- x.seq
      dfg$Frequency <- log1p(dfg$Frequency)
      
      X_raw    <- as.matrix(dfg[, features])
      X_scaled <- sweep(X_raw,  2, scaler$feature_means, "-")
      X_scaled <- sweep(X_scaled, 2, scaler$feature_sds,   "/")
      
      # extract parameters for this draw
      W1 <- get_param(draws_mat, i, "W1", c(H1, length(features)))
      b1 <- get_param(draws_mat, i, "b1", H1)
      W2 <- get_param(draws_mat, i, "W2", c(H2, H1))
      b2 <- get_param(draws_mat, i, "b2", H2)
      W3 <- get_param(draws_mat, i, "W3", c(H3, H2))
      b3 <- get_param(draws_mat, i, "b3", H3)
      w4 <- get_param(draws_mat, i, "w4", H3)
      b4 <- get_param(draws_mat, i, "b4", 1)
      
      A1        <- tanh(W1 %*% t(X_scaled) + b1)
      A2        <- tanh(W2 %*% A1         + b2)
      A3        <- tanh(W3 %*% A2         + b3)
      mu_scaled <- as.numeric(crossprod(w4, A3) + b4)
      mu        <- mu_scaled * scaler$target_sd + scaler$target_mean
      
      tibble::tibble(
        draw      = i,
        Feature   = feat,
        Value     = x.seq,
        Predicted = mu
      )
    })
  })
  
  # ---------------- mean & 90 % CI curves --------------------------
  stats_df <- samples_df %>%
    group_by(Feature, Value) %>%
    summarise(
      MeanPred = mean(Predicted),
      Lower90  = quantile(Predicted, 0.05),
      Upper90  = quantile(Predicted, 0.95),
      .groups  = "drop"
    )
  
  # ---------------- raw scatter ------------------------------------
  scatter_df <- df_raw %>%
    rename(Actual = SoundPressure) %>%
    pivot_longer(all_of(features),
                 names_to  = "Feature",
                 values_to = "Value")
  
  ggplot() +
    geom_point(
      data  = scatter_df,
      aes(x = Value, y = Actual, colour = "Actual data"),
      size  = 0.5,
      alpha = 0.2
    ) +
    geom_line(
      data  = samples_df,
      aes(x = Value, y = Predicted,
          group = interaction(draw, Feature),
          colour = "Posterior samples"),
      alpha = 0.1,
      size  = 0.2
    ) +
    geom_line(                            # posterior mean
      data  = stats_df,
      aes(x = Value, y = MeanPred, colour = "Posterior mean"),
      size  = 1
    ) +
    geom_line(                            # 90 % upper
      data  = stats_df,
      aes(x = Value, y = Upper90, colour = "90% CI upper"),
      size  = 0.5,
      linetype = "dashed"
    ) +
    geom_line(                            # 90 % lower
      data  = stats_df,
      aes(x = Value, y = Lower90, colour = "90% CI lower"),
      size  = 0.5,
      linetype = "dashed"
    ) +
    facet_wrap(~ Feature, scales = "free_x") +
    theme_minimal()
}


darken_color <- function(col, amount = 0.3) {
  # col2rgb returns 0–255
  rgb_vals <- grDevices::col2rgb(col)
  # scale down each channel
  rgb_new  <- pmax(rgb_vals * (1 - amount), 0)
  # reassemble into a hex string
  grDevices::rgb(
    red   = rgb_new[1, ] / 255,
    green = rgb_new[2, ] / 255,
    blue  = rgb_new[3, ] / 255
  )
}


plot_pd_credible_band <- function(
    draw_files,          # character vector of your *_MCMC_draws.rds paths
    dataset_rds_path,    # "data/uci/airfoil_dataset.rds"
    scaler_rds_path,     # "data/uci/airfoil_scaler.rds"
    H1, H2, H3,          # your hidden layer sizes
    level   = 0.90,      # credible level (0.90 = 90% band)
    n_grid  = 100        # points per feature
) {
  library(posterior)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  
  # --- 1) load & bind chains into a single draws_array
  draws1 <- load_draws(draw_files[[1]])
  draws2 <- load_draws(draw_files[[2]])
  
  df1 <- as_draws_df(draws1)
  df1$.chain     <- 1L
  df1$.iteration <- seq_len(nrow(df1))
  
  df2 <- as_draws_df(draws2)
  df2$.chain     <- 2L
  df2$.iteration <- seq_len(nrow(df2))
  
  df_both <- rbind(df1, df2)
  both    <- as_draws_array(df_both)
  
  mat <- as_draws_matrix(both)   # (iter×chains) × variables
  cn  <- colnames(mat)
  
  # helper to extract either scalars, vectors, or matrices
  get_vec <- function(i, prefix, dims) {
    if (length(dims) == 1) {
      if (prefix %in% cn) {
        return(mat[i, prefix])
      }
      cols <- paste0(prefix, "[", seq_len(dims), "]")
      return(as.numeric(mat[i, cols]))
    }
    # matrix case
    rows <- dims[1]; cols <- dims[2]
    names <- as.vector(outer(
      seq_len(rows), seq_len(cols),
      function(r, c) sprintf("%s[%d,%d]", prefix, r, c)
    ))
    return(matrix(mat[i, names], nrow = rows, byrow = FALSE))
  }
  
  # --- 2) load raw data & scaler
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  # --- 3) build partial-dependence bands
  pd_list <- lapply(features, function(feat) {
    x_seq <- seq(
      min(df_raw[[feat]]),
      max(df_raw[[feat]]),
      length.out = n_grid
    )
    
    # matrix of (total_draws) × (n_grid) predictions
    pd_mat <- sapply(x_seq, function(xj) {
      # construct feature vector at xj, others at median
      x0 <- sapply(features, function(f) {
        if (f == feat) xj else median(df_raw[[f]])
      })
      names(x0) <- features
      
      # log-transform & scale as vector
      x0["Frequency"] <- log1p(x0["Frequency"])
      x_sc <- (x0 - scaler$feature_means) / scaler$feature_sds
      
      # forward pass for each posterior draw
      sapply(seq_len(nrow(mat)), function(i) {
        W1 <- get_vec(i, "W1", c(H1, length(features)))
        b1 <- get_vec(i, "b1", H1)
        W2 <- get_vec(i, "W2", c(H2, H1))
        b2 <- get_vec(i, "b2", H2)
        W3 <- get_vec(i, "W3", c(H3, H2))
        b3 <- get_vec(i, "b3", H3)
        w4 <- get_vec(i, "w4", H3)
        b4 <- get_vec(i, "b4", 1)
        
        a1 <- tanh(W1 %*% x_sc + b1)
        a2 <- tanh(W2 %*% a1   + b2)
        a3 <- tanh(W3 %*% a2   + b3)
        as.numeric(w4 %*% a3  + b4)
      })
    })
    
    # summarize quantiles across draws
    tibble(
      Feature = feat,
      Value   = x_seq,
      qlow    = apply(pd_mat, 2, quantile, (1 - level) / 2),
      qmid    = apply(pd_mat, 2, quantile, 0.5),
      qhi     = apply(pd_mat, 2, quantile, 1 - (1 - level) / 2)
    )
  })
  
  pd_all <- bind_rows(pd_list)
  
  # --- 4) plot
  ggplot(pd_all, aes(x = Value)) +
    geom_ribbon(aes(ymin = qlow, ymax = qhi),
                fill = "steelblue", alpha = 0.3) +
    geom_line(aes(y = qmid), color = "steelblue", size = 1) +
    facet_wrap(~ Feature, scales = "free_x") +
    labs(
      title = sprintf("%.1f%% Credible Partial‐Dependence", level * 100),
      x     = "Feature value",
      y     = "Predicted SoundPressure"
    ) +
    theme_minimal()
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility: load & combine an arbitrary number of chain files into a matrix
# draw_files: character vector of paths to your *_MCMC_draws.rds
# returns: a matrix (draws × parameters)
load_and_combine <- function(draw_files) {
  dfs <- lapply(seq_along(draw_files), function(i) {
    arr <- load_draws(draw_files[i])
    df  <- as_draws_df(arr)
    df$.chain     <- i
    df$.iteration <- seq_len(nrow(df))
    df
  })
  arr_all <- as_draws_array(do.call(rbind, dfs))
  as_draws_matrix(arr_all)
}

plot_pd_credible_band <- function(
    draw_files,         # vector of *_MCMC_draws.rds paths (length ≥ 1)
    dataset_rds_path,   # "data/uci/airfoil_dataset.rds"
    scaler_rds_path,    # "data/uci/airfoil_scaler.rds"
    H1, H2, H3,         # hidden layer sizes
    level = 0.90,       # credible level
    n_grid = 100        # points per feature
) {
  # 1) load & combine all chains
  mat <- load_and_combine(draw_files)
  cn  <- colnames(mat)
  
  # helper: extract parameter slices
  get_vec <- function(i, prefix, dims) {
    if (length(dims) == 1) {
      # scalar or vector
      if (dims == 1 && prefix %in% cn) return(mat[i, prefix])
      cols <- paste0(prefix, "[", seq_len(dims), "]")
      return(as.numeric(mat[i, cols]))
    }
    # matrix case
    rows <- dims[1]; cols <- dims[2]
    names <- as.vector(outer(seq_len(rows), seq_len(cols),
                             function(r,c) sprintf("%s[%d,%d]", prefix, r, c)))
    matrix(mat[i, names], nrow = rows, byrow = FALSE)
  }
  
  # 2) raw data & scaler
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  # 3) build credible‐band for each feature
  pd_list <- lapply(features, function(feat) {
    x_seq <- seq(min(df_raw[[feat]]), max(df_raw[[feat]]),
                 length.out = n_grid)
    # matrix draws × grid
    pd_mat <- sapply(x_seq, function(xj) {
      # build input: this feat = xj, others = median(raw)
      x0 <- sapply(features, function(f)
        if (f==feat) xj else median(df_raw[[f]]))
      names(x0) <- features
      # log-transform + scale
      x0["Frequency"] <- log1p(x0["Frequency"])
      x_sc <- (x0 - scaler$feature_means) / scaler$feature_sds
      
      # forward pass for each draw
      sapply(seq_len(nrow(mat)), function(i) {
        W1 <- get_vec(i, "W1", c(H1, length(features)))
        b1 <- get_vec(i, "b1", H1)
        W2 <- get_vec(i, "W2", c(H2, H1))
      b2 <- get_vec(i, "b2", H2)
      W3 <- get_vec(i, "W3", c(H3, H2))
    b3 <- get_vec(i, "b3", H3)
    w4 <- get_vec(i, "w4", H3)
    b4 <- get_vec(i, "b4", 1)
    a1 <- tanh(W1 %*% x_sc + b1)
    a2 <- tanh(W2 %*% a1    + b2)
    a3 <- tanh(W3 %*% a2    + b3)
    as.numeric(w4 %*% a3   + b4)
      })
    })
tibble(
  Feature = feat,
  Value   = x_seq,
  qlow    = apply(pd_mat, 2, quantile, (1-level)/2),
  qmid    = apply(pd_mat, 2, quantile, 0.5),
  qhi     = apply(pd_mat, 2, quantile, 1-(1-level)/2)
)
  })
pd_all <- bind_rows(pd_list)

# 4) plot
ggplot(pd_all, aes(x = Value)) +
  geom_ribbon(aes(ymin = qlow, ymax = qhi),
              fill = "steelblue", alpha = 0.3) +
  geom_line(aes(y = qmid), color = "steelblue", size = 1) +
  facet_wrap(~ Feature, scales = "free_x") +
  labs(
    title = sprintf("%.1f%% Credible Partial‐Dependence", level*100),
    x = "Feature value",
    y = "Predicted SoundPressure"
  ) +
  theme_minimal()
}


# Overlay traceplot for any number of chains
traceplot_chains <- function(draws, param) {
  # draws: either
  #  • a character vector of file paths to *_MCMC_draws.rds, or
  #  • a posterior::draws_array / draws_df with .chain and .iteration, or
  #  • a data.frame already in draws_df form
  
  # 1) Build a single data.frame with columns .chain, .iteration, and all parameters
  df_all <- NULL
  if (is.character(draws)) {
    dfs <- lapply(seq_along(draws), function(i) {
      df_i <- as_draws_df(load_draws(draws[i]))
      df_i$.chain     <- i
      df_i$.iteration <- seq_len(nrow(df_i))
      df_i
    })
    df_all <- do.call(rbind, dfs)
  } else if (inherits(draws, "draws_array")) {
    df_all <- as_draws_df(draws)
    if (!(".chain" %in% names(df_all))) stop("draws_array must include .chain")
    if (!(".iteration" %in% names(df_all))) {
      df_all$.iteration <- rep(seq_len(nrow(df_all)/length(unique(df_all$.chain))),
                               times = length(unique(df_all$.chain)))
    }
  } else if (is.data.frame(draws) &&
             all(c(".chain", ".iteration") %in% names(draws))) {
    df_all <- draws
  } else {
    stop("Unsupported input. Provide file paths or a draws_array/draws_df.")
  }
  
  # 2) Check parameter exists
  if (!param %in% colnames(df_all)) {
    stop("Parameter '", param, "' not found in draws.")
  }
  
  # 3) Plot
  ggplot(df_all, aes(x = .iteration, y = .data[[param]], color = factor(.chain))) +
    geom_line(size = 1) +
    scale_color_manual(
      name   = "Chain",
      values = c(
        "1" = "#1B9E77",
        "2" = "#D95F02",
        "3" = "#7570B3",
        "4" = "#E7298A"
      )
    ) +
    labs(
      title = paste("Traceplot of", param),
      x     = "Iteration",
      y     = param
    ) +
    theme_minimal()
}