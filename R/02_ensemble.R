# 02_ensemble.R
# Modular Deep Ensemble Training & Prediction Pipeline

# Install/load required packages
if (!requireNamespace("keras", quietly = TRUE)) {
  install.packages("keras")
  keras::install_keras(envname = "r-tensorflow", method = "virtualenv")
}
library(keras)
library(purrr)
library(ggplot2)
library(dplyr)

# Load all .rds datasets from a folder
load_datasets <- function(
    data_path = "data/synthetic"
) {
  files <- list.files(data_path, pattern = "_dataset\\.rds$", full.names = TRUE)
  data_list <- map(files, readRDS)
  names(data_list) <- tools::file_path_sans_ext(basename(files))
  return(data_list)
}

# Build a fresh MLP model
build_model <- function(
    input_dim,
    hidden_units = c(64, 64, 32),
    activation   = "tanh",
    output_units = 1
) {
  model <- keras_model_sequential()
  # first hidden layer with input shape
  model <- model %>%
    layer_dense(units = hidden_units[1], activation = activation,
                input_shape = input_dim)
  # additional hidden layers
  if (length(hidden_units) > 1) {
    for (h in hidden_units[-1]) {
      model <- model %>% layer_dense(units = h, activation = activation)
    }
  }
  # output layer
  model <- model %>% layer_dense(units = output_units, activation = "linear")
  # compile
  model %>% compile(
    optimizer = optimizer_adam(),
    loss      = "mse",
    metrics   = list("mean_squared_error")
  )
  return(model)
}

# Train an ensemble on one dataset
train_ensemble_for_dataset <- function(
    df,
    dataset_name,
    ensemble_size = 4,
    epochs        = 100,
    batch_size    = 32,
    hidden_units  = c(64, 64, 32),
    activation    = "tanh",
    output_units  = 1,
    save_path     = "results/ensemble"
) {
  dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  x <- as.matrix(df$x)
  y <- df$y
  input_dim <- ncol(x)
  for (m in seq_len(ensemble_size)) {
    set.seed(123 + m)
    model <- build_model(
      input_dim    = input_dim,
      hidden_units = hidden_units,
      activation   = activation,
      output_units = output_units
    )
    model %>% fit(
      x = x, y = y,
      epochs     = epochs,
      batch_size = batch_size,
      verbose    = 0
    )
    fname <- file.path(save_path,
                       sprintf("%s_member%02d.keras", dataset_name, m))
    save_model_tf(model, fname)
    message("Saved ensemble member: ", fname)
  }
}

# Master pipeline over multiple datasets
run_ensemble_pipeline <- function(
    data_path     = "data/synthetic",
    ensemble_size = 4,
    epochs        = 100,
    batch_size    = 32,
    hidden_units  = c(64, 64, 32),
    activation    = "tanh",
    output_units  = 1,
    save_path     = "results/ensemble_synth"
) {
  datasets <- load_datasets(data_path)
  walk2(datasets, names(datasets), ~ {
    message("Training ensemble for dataset: ", .y)
    train_ensemble_for_dataset(
      df            = .x,
      dataset_name  = .y,
      ensemble_size = ensemble_size,
      epochs        = epochs,
      batch_size    = batch_size,
      hidden_units  = hidden_units,
      activation    = activation,
      output_units  = output_units,
      save_path     = save_path
    )
  })
}

# Predict & visualize for one model
predict_x_vs_y <- function(
    model_path,
    data_path
) {
  model <- load_model_tf(model_path)
  df <- readRDS(data_path)
  x <- df$x
  y_true <- df$y
  y_pred <- as.numeric(model %>% predict(matrix(x, ncol = 1)))
  plot_df <- tibble(x = x, actual = y_true, predicted = y_pred) %>%
    arrange(x)
  ggplot(plot_df, aes(x = x)) +
    geom_point(aes(y = actual, color = "Actual"), alpha = 0.25, size = 0.7) +
    geom_line(aes(y = predicted, color = "Predicted"), size = 1) +
    geom_line(aes(y = sin(2 * pi * x / 2), color = "True Process"),
              size = 1, linetype = "dashed") +
    scale_color_manual(
      name = NULL,
      values = c(
        Actual        = "blue",
        Predicted     = "orange",
        "True Process" = "red"
      )
    ) +
    labs(title = basename(model_path), x = "x", y = "y value") +
    theme_minimal()
}


plot_predict_x_vs_y_facets <- function(
    model_paths,  # character vector of paths to saved .keras models
    data_path,    # path to the .rds containing a data.frame/tibble with columns x and y
    ncol = 2      # number of columns in the facet grid
) {
  library(keras)
  library(dplyr)
  library(purrr)
  library(ggplot2)
  
  # load the raw data once
  df <- readRDS(data_path)
  
  # assemble predictions + labels
  all_preds <- map_dfr(model_paths, function(path) {
    m       <- load_model_tf(path)
    x       <- df$x
    y_obs   <- df$y
    y_hat   <- as.numeric(m %>% predict(matrix(x, ncol = 1)))
    fname   <- basename(path)
    member  <- sub(".*member([0-9]{2})_canon\\.keras$", "\\1", fname)
    tibble(
      x           = x,
      actual      = y_obs,
      predicted   = y_hat,
      facet_label = paste0("Simulation Study, Ensemble Member ", member)
    )
  }) %>% arrange(facet_label, x)
  
  # plot
  ggplot(all_preds, aes(x = x)) +
    geom_point(aes(y = actual, color = "Actual"),
               alpha = 0.25, size = 0.7) +
    geom_line(aes(y = predicted, color = "Predicted"),
              size = 1) +
    geom_line(aes(y = sin(2 * pi * x / 2), color = "True Process"),
              size = 1, linetype = "dashed") +
    scale_color_manual(
      name   = NULL,
      values = c(
        Actual        = "blue",
        Predicted     = "orange",
        "True Process" = "red"
      )
    ) +
    facet_wrap(~ facet_label, ncol = ncol) +
    labs(x = "x", y = "y value") +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 10, face = "bold")
    )
}


#UCI Airfoil specific functions

# 1) Download the Airfoil Self-Noise dataset
data_dir_default <- "data/uci"
load_airfoil_data <- function(data_dir = data_dir_default) {
  dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
  colnames <- c("Frequency", "AngleAttack", "ChordLength",
                "Velocity", "SuctionThickness", "SoundPressure")
  df <- read.table(url, col.names = colnames)
  saveRDS(df, file.path(data_dir, "airfoil_dataset.rds"))
  message("Airfoil data saved to: ", file.path(data_dir, "airfoil_dataset.rds"))
  invisible(df)
}

# ──────────────────────────────────────────────────────────────────────────────
# 1) Prepare & save (features + optional target) with z-scaling
prepare_and_save_airfoil <- function(input_file,
                                     output_file,
                                     scaler_file,
                                     scale_target = TRUE) {
  # load
  df_raw <- readRDS(input_file)
  # log-transform Frequency
  df_raw$Frequency <- log1p(df_raw$Frequency)
  
  # define
  features <- c("Frequency",
                "AngleAttack",
                "ChordLength",
                "Velocity",
                "SuctionThickness")
  
  # compute feature means/sds
  feat_means <- vapply(df_raw[features], mean, numeric(1), na.rm = TRUE)
  feat_sds   <- vapply(df_raw[features], sd,   numeric(1), na.rm = TRUE)
  
  # scale features
  df_scaled <- df_raw
  df_scaled[features] <- scale(df_raw[features],
                               center = feat_means,
                               scale  = feat_sds)
  
  # optionally scale target
  if (scale_target) {
    target_mean <- mean(df_raw$SoundPressure, na.rm = TRUE)
    target_sd   <- sd(df_raw$SoundPressure,   na.rm = TRUE)
    df_scaled$SoundPressure <- scale(df_raw$SoundPressure,
                                     center = target_mean,
                                     scale  = target_sd)
  }
  
  # save
  saveRDS(df_scaled, output_file)
  message("Preprocessed Airfoil data saved to: ", output_file)
  
  # write scaler
  scaler <- list(
    feature_means = feat_means,
    feature_sds   = feat_sds
  )
  if (scale_target) {
    scaler$target_mean <- target_mean
    scaler$target_sd   <- target_sd
  }
  saveRDS(scaler, scaler_file)
  message("Scaler parameters saved to: ", scaler_file)
  
  invisible(scaler)
}


# ──────────────────────────────────────────────────────────────────────────────
# 2) Train a simple deep-ensemble (on already-scaled data)
train_airfoil_ensemble <- function(
    dataset_rds_path,
    ensemble_size  = 4,
    epochs         = 400,
    batch_size     = 32,
    hidden_units   = c(64, 64, 32),
    activation     = "tanh",
    output_units   = 1,
    save_path      = "results/ensemble_airfoil",
    val_split      = 0.2,
    patience       = 20
) {
  dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  df <- readRDS(dataset_rds_path)
  
  features <- c("Frequency", "AngleAttack", "ChordLength", "Velocity", "SuctionThickness")
  x <- as.matrix(df[, features])
  y <- df$SoundPressure
  
  for (m in seq_len(ensemble_size)) {
    set.seed(123 + m)
    model <- build_model(
      input_dim    = ncol(x),
      hidden_units = hidden_units,
      activation   = activation,
      output_units = output_units
    )
    
    history <- model %>% fit(
      x               = x, 
      y               = y,
      validation_split= val_split,
      epochs          = epochs,
      batch_size      = batch_size,
      callbacks       = list(
        callback_early_stopping(
          monitor = "val_loss",
          patience = patience,
          restore_best_weights = TRUE
        )
      ),
      verbose = 0
    )
    
    fname <- file.path(save_path, sprintf("airfoil_member%02d.keras", m))
    save_model_tf(model, fname)
    message("Member ", m, " stopped at epoch ", 
            which.min(history$metrics$val_loss), 
            " (of max ", epochs, "). ",
            "Saved to ", fname)
  }
}


# ──────────────────────────────────────────────────────────────────────────────
# 3) Compute raw+predictions, undoing target-scaling if needed
compute_airfoil_preds <- function(
    model_path,
    dataset_rds_path,
    scaler_rds_path
) {
  library(keras); library(dplyr)
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  # apply log1p + feature scaling
  df_proc <- df_raw
  df_proc$Frequency <- log1p(df_proc$Frequency)
  x_raw    <- as.matrix(df_proc[, features])
  x_scaled <- sweep(x_raw,  2, scaler$feature_means, "-")
  x_scaled <- sweep(x_scaled, 2, scaler$feature_sds,   "/")
  
  # predict
  model <- load_model_tf(model_path)
  preds  <- as.numeric(model %>% predict(x_scaled))
  
  # undo target scaling
  if (!is.null(scaler$target_mean)) {
    preds <- preds * scaler$target_sd + scaler$target_mean
  }
  
  bind_cols(df_raw, Predicted = preds)
}


# ──────────────────────────────────────────────────────────────────────────────
# 4) Single-feature scatter + sorted line (optional)
predict_airfoil_vs_y <- function(
    model_path,
    dataset_rds_path,
    scaler_rds_path,
    feature = "Frequency"
) {
  library(ggplot2); library(dplyr)
  
  dfp <- compute_airfoil_preds(model_path, dataset_rds_path, scaler_rds_path)
  dfp <- dfp %>% arrange(.data[[feature]])
  
  ggplot(dfp, aes(x = .data[[feature]])) +
    geom_point(aes(y = SoundPressure, color = "Actual"),
               size = 1, alpha = 0.6) +
    geom_line(aes(y = Predicted,   color = "Predicted",
                  group = 1),       # force single line
              size = 0.8) +
    scale_color_manual(NULL,
                       values = c(Actual = "blue", Predicted = "red")) +
    labs(
      title = paste(basename(model_path), "–", feature),
      x     = feature,
      y     = "SoundPressure"
    ) +
    theme_minimal()
}


# ──────────────────────────────────────────────────────────────────────────────
# 5) Partial‐dependence facets: vary one feature at a time,
#    hold all others at their (log‐transformed) mean, and draw a smooth curve
predict_airfoil_vs_y_all_features <- function(
    model_path,
    dataset_rds_path,
    scaler_rds_path,
    n.grid = 100
) {
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(purrr)
  
  # raw + scaler
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  # 1) compute mean of every feature *after* log1p on Frequency
  df_means <- df_raw %>%
    mutate(Frequency = log1p(Frequency)) %>%
    summarise(across(all_of(features), mean, na.rm=TRUE))
  
  # 2) build a grid + predict for each feature
  pd <- map_df(features, function(feat) {
    x.seq <- seq(min(df_raw[[feat]], na.rm=TRUE),
                 max(df_raw[[feat]], na.rm=TRUE),
                 length.out = n.grid)
    # replicate the mean row n.grid times
    dfg <- df_means[rep(1, n.grid), ] %>% as.data.frame()
    dfg[[feat]] <- x.seq
    # log + scale
    dfg$Frequency <- log1p(dfg$Frequency)
    x.raw    <- as.matrix(dfg[, features])
    x.scaled <- sweep(x.raw,  2, scaler$feature_means, "-")
    x.scaled <- sweep(x.scaled, 2, scaler$feature_sds,   "/")
    # predict + undo target scaling
    model <- load_model_tf(model_path)
    y.pred <- as.numeric(model %>% predict(x.scaled))
    if (!is.null(scaler$target_mean)) {
      y.pred <- y.pred * scaler$target_sd + scaler$target_mean
    }
    tibble(Feature   = feat,
           Value     = x.seq,
           Predicted = y.pred)
  })
  
  # 3) pivot raw for scatter
  raw_long <- df_raw %>%
    rename(Actual = SoundPressure) %>%
    mutate(Frequency = Frequency) %>%
    pivot_longer(all_of(features),
                 names_to  = "Feature",
                 values_to = "Value")
  
  # 4) plot
  ggplot() +
    geom_point(data = raw_long,
               aes(x = Value, y = Actual),
               color = "blue", size = 0.6, alpha = 0.4) +
    geom_line(data = pd,
              aes(x = Value, y = Predicted),
              color = "red", size = 1) +
    facet_wrap(~ Feature, scales = "free_x") +
    labs(
      title = paste("Partial Dependence –", basename(model_path)),
      x     = "Feature value",
      y     = "SoundPressure"
    ) +
    theme_minimal()
}

compare_partial_dependence_models <- function(
    model_paths,
    model_labels = basename(model_paths),
    dataset_rds_path,
    scaler_rds_path,
    n.grid = 100
) {
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  
  # 1) load raw data & scaler
  df_raw <- readRDS(dataset_rds_path)
  scaler <- readRDS(scaler_rds_path)
  features <- names(scaler$feature_means)
  
  # 2) compute feature‐means (after log1p on Frequency) for grid baseline
  df_means <- df_raw %>%
    mutate(Frequency = log1p(Frequency)) %>%
    summarise(across(all_of(features), mean, na.rm = TRUE))
  
  # 3) build a big data.frame of predictions:
  pd_all <- map2_dfr(model_paths, model_labels, function(path, lbl) {
    model <- load_model_tf(path)
    map_df(features, function(feat) {
      # make a grid on the *raw* feature scale
      x.seq <- seq(min(df_raw[[feat]], na.rm=TRUE),
                   max(df_raw[[feat]], na.rm=TRUE),
                   length.out = n.grid)
      # replicate the mean‐row and swap in this feature’s grid
      dfg <- as.data.frame(df_means[rep(1, n.grid), ])
      dfg[[feat]] <- x.seq
      # apply the same log1p+scaling on Frequency
      dfg$Frequency <- log1p(dfg$Frequency)
      x.raw    <- as.matrix(dfg[, features])
      x.scaled <- sweep(x.raw, 2, scaler$feature_means, "-")
      x.scaled <- sweep(x.scaled, 2, scaler$feature_sds,   "/")
      # predict & undo target scaling if needed
      ypred <- as.numeric(model %>% predict(x.scaled))
      if (!is.null(scaler$target_mean)) {
        ypred <- ypred * scaler$target_sd + scaler$target_mean
      }
      tibble(
        Feature   = feat,
        Value     = x.seq,
        Predicted = ypred,
        Model     = lbl
      )
    })
  })
  
  # 4) pivot the raw data for scatter
  raw_long <- df_raw %>%
    rename(Actual = SoundPressure) %>%
    mutate(Frequency = Frequency) %>%
    pivot_longer(all_of(features),
                 names_to  = "Feature",
                 values_to = "Value")
  
  # 5) plot them all together
  ggplot() +
    geom_point(data = raw_long,
               aes(x = Value, y = Actual),
               color = "blue", size = 0.5, alpha = 0.2) +
    geom_line(data = pd_all,
              aes(x = Value, y = Predicted, color = Model),
              size = 1) +
    facet_wrap(~ Feature, scales = "free_x", ncol = 4) +
    scale_color_brewer("Model", palette = "Dark2") +
    labs(
      title = "UCI Airfoil: Partial Dependence Comparison between all 4 NNs",
      x     = "Feature value",
      y     = "SoundPressure"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
}



# ──────────────────────────────────────────────────────────────────────────────

