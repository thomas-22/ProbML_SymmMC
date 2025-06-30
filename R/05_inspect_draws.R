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
               size = 0.6, alpha = 0.3, color = "grey50") +
    geom_line(data = samples_df,
              aes(x = Value, y = Predicted, group = draw),
              alpha = 0.3, color = "darkgreen") +
    facet_wrap(~ Feature, scales = "free_x") +
    labs(
      title = "Airfoil DEI-MCMC: Partial Dependence (Posterior Samples)",
      x     = "Feature value",
      y     = "SoundPressure"
    ) +
    theme_minimal()
}

# Reads a text file of keras model paths, overlays their PD curves on the posterior-sample plot
plot_posterior_ensemble_and_dei <- function(
    draw_file,
    nn_paths_file,
    dataset_rds_path = "data/uci/airfoil_dataset.rds",
    scaler_rds_path  = "data/uci/airfoil_scaler.rds",
    H1, H2, H3,
    n.grid    = 100,
    num_draws = 20,
    seed      = 42
) {
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(keras)
  
  # base posterior-sample plot
  p_base <- plot_posterior_predictive_samples_uciairfoil(
    draw_file         = draw_file,
    dataset_rds_path  = dataset_rds_path,
    scaler_rds_path   = scaler_rds_path,
    H1 = H1, H2 = H2, H3 = H3,
    n.grid    = n.grid,
    num_draws = num_draws,
    seed      = seed
  )
  
  # load raw & scaler
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  raw_means <- df_raw %>%
    mutate(Frequency = log1p(Frequency)) %>%
    summarise(across(all_of(features), mean, na.rm=TRUE))
  
  # read ensemble Keras paths
  nn_paths <- readLines(nn_paths_file)
  
  # build ensemble PD data
  ens_df <- map_dfr(nn_paths, function(mp) {
    model <- load_model_tf(mp)
    map_dfr(features, function(feat) {
      x.seq <- seq(min(df_raw[[feat]], na.rm=TRUE),
                   max(df_raw[[feat]], na.rm=TRUE),
                   length.out = n.grid)
      dfg <- raw_means[rep(1, n.grid), ] %>% as.data.frame()
      dfg[[feat]] <- x.seq
      dfg$Frequency <- log1p(dfg$Frequency)
      X_raw <- as.matrix(dfg[, features])
      X_scaled <- sweep(X_raw, 2, scaler$feature_means, "-")
      X_scaled <- sweep(X_scaled, 2, scaler$feature_sds,   "/")
      preds <- as.numeric(model %>% predict(X_scaled))
      preds <- preds * scaler$target_sd + scaler$target_mean
      tibble(
        Model     = basename(mp),
        Feature   = feat,
        Value     = x.seq,
        Predicted = preds
      )
    })
  })
  
  # overlay ensemble lines
  p_base +
    geom_line(
      data = ens_df,
      aes(x = Value, y = Predicted, color = Model),
      size = 1
    ) +
    scale_color_brewer(palette = "Dark2", name = "Ensemble") +
    labs(
      title = "Airfoil PD: DEI Samples + Ensemble NNs",
      x     = "Feature value",
      y     = "SoundPressure"
    ) +
    theme(legend.position = "bottom")
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