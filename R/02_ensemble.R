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
    geom_point(aes(y = actual, color = "Actual"), alpha = 0.6) +
    geom_line(aes(y = predicted, color = "Predicted"), size = 1) +
    geom_line(aes(y = sin(2 * pi * x / 2), color = "True Process"),
              size = 1, linetype = "dashed") +
    scale_color_manual(
      name = NULL,
      values = c(
        Actual        = "blue",
        Predicted     = "red",
        "True Process" = "green"
      )
    ) +
    labs(title = basename(model_path), x = "x", y = "y value") +
    theme_minimal()
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

# Prepare the dataset
prepare_and_save_airfoil <- function(input_file,
                                     output_file,
                                     scale_target = FALSE) {
  # 1) Rohdaten laden
  df_raw <- readRDS(input_file)
  
  # 2) Log-Transform für Frequency (vermeidet Ausreißer)
  df_raw$Frequency <- log1p(df_raw$Frequency)
  
  # 3) Features definieren
  features <- c("Frequency",
                "AngleAttack",
                "ChordLength",
                "Velocity",
                "SuctionThickness")
  
  # 4) Skalen für Features berechnen
  feat_means <- vapply(df_raw[features], mean, numeric(1), na.rm = TRUE)
  feat_sds   <- vapply(df_raw[features], sd,   numeric(1), na.rm = TRUE)
  
  # 5) DataFrame kopieren und Features z-standardisieren
  df_scaled <- df_raw
  df_scaled[features] <- scale(df_raw[features],
                               center = feat_means,
                               scale  = feat_sds)
  
  # 6) Optional: auch SoundPressure skalieren
  if (scale_target) {
    target_mean <- mean(df_raw$SoundPressure, na.rm = TRUE)
    target_sd   <- sd(df_raw$SoundPressure,   na.rm = TRUE)
    df_scaled$SoundPressure <- scale(df_raw$SoundPressure,
                                     center = target_mean,
                                     scale  = target_sd)
  }
  
  # 7) Speichern
  saveRDS(df_scaled, output_file)
  message("Preprocessed Airfoil data saved to: ", output_file)
  
  # 8) Invisibly: Rückgabe der Skalierungsparameter
  out <- list(
    feature_means = feat_means,
    feature_sds   = feat_sds
  )
  if (scale_target) {
    out$target_mean <- target_mean
    out$target_sd   <- target_sd
  }
  invisible(out)
}



# 2) Train a deep ensemble on the Airfoil dataset with input scaling
train_airfoil_ensemble <- function(
    data_dir       = data_dir_default,
    ensemble_size  = 4,
    epochs         = 200,
    batch_size     = 32,
    hidden_units   = c(64, 64, 32),
    activation     = "tanh",
    output_units   = 1,
    save_path      = "results/ensemble_airfoil"
) {
  dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  df <- readRDS(file.path(data_dir, "airfoil_dataset.rds"))
  features <- c("Frequency", "AngleAttack", "ChordLength", "Velocity", "SuctionThickness")
  x_raw <- as.matrix(df[, features])
  y     <- df$SoundPressure
  
  # compute scaling parameters
  mu    <- colMeans(x_raw)
  sigma <- apply(x_raw, 2, sd)
  x_scaled <- scale(x_raw, center = mu, scale = sigma)
  
  # save scaler for prediction
  saveRDS(list(mu = mu, sigma = sigma), file.path(data_dir, "airfoil_scaler.rds"))
  message("Saved scaler to: ", file.path(data_dir, "airfoil_scaler.rds"))
  
  input_dim <- ncol(x_scaled)
  for (m in seq_len(ensemble_size)) {
    set.seed(123 + m)
    model <- build_model(
      input_dim    = input_dim,
      hidden_units = hidden_units,
      activation   = activation,
      output_units = output_units
    )
    model %>% fit(
      x = x_scaled, y = y,
      epochs     = epochs,
      batch_size = batch_size,
      verbose    = 0
    )
    fname <- file.path(save_path,
                       sprintf("airfoil_member%02d.keras", m))
    save_model_tf(model, fname)
    message("Saved Airfoil ensemble member: ", fname)
  }
}

# 3) Helper to load & scale Airfoil inputs for prediction
load_airfoil_scaled_inputs <- function(data_dir = data_dir_default) {
  df <- readRDS(file.path(data_dir, "airfoil_dataset.rds"))
  features <- c("Frequency", "AngleAttack", "ChordLength", "Velocity", "SuctionThickness")
  x_raw <- as.matrix(df[, features])
  scaler <- readRDS(file.path(data_dir, "airfoil_scaler.rds"))
  x_scaled <- sweep(x_raw, 2, scaler$mu, "-")
  x_scaled <- sweep(x_scaled, 2, scaler$sigma, "/")
  list(df = df, x_scaled = x_scaled)
}

# 4) Predict & visualize on one feature
predict_airfoil_vs_y <- function(
    model_path,
    data_dir = data_dir_default,
    feature  = "Frequency"
) {
  info     <- load_airfoil_scaled_inputs(data_dir)
  df       <- info$df
  x_scaled <- info$x_scaled
  model    <- load_model_tf(model_path)
  
  y_pred <- as.numeric(model %>% predict(x_scaled))
  plot_df <- tibble(
    x         = df[[feature]],
    actual    = df$SoundPressure,
    predicted = y_pred
  ) %>% arrange(x)
  
  ggplot(plot_df, aes(x = x)) +
    geom_point(aes(y = actual, color = "Actual"), alpha = 0.6, size = 1) +
    geom_line(aes(y = predicted, color = "Predicted"), size = 0.8) +
    labs(
      title = paste(basename(model_path), "on Airfoil:", feature),
      x = feature, y = "SoundPressure"
    ) +
    scale_color_manual(name = NULL,
                       values = c(Actual = "blue", Predicted = "red")) +
    theme_minimal()
}

# 5) Predict & visualize over all features with facets
predict_airfoil_vs_y_all_features <- function(
    model_path,
    data_dir = data_dir_default
) {
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  
  info     <- load_airfoil_scaled_inputs(data_dir)
  df       <- info$df
  x_scaled <- info$x_scaled
  model    <- load_model_tf(model_path)
  
  features <- c("Frequency", "AngleAttack", "ChordLength", "Velocity", "SuctionThickness")
  y_pred   <- as.numeric(model %>% predict(x_scaled))
  
  df_long <- df %>%
    mutate(Predicted = y_pred) %>%
    select(all_of(features), SoundPressure, Predicted) %>%
    pivot_longer(
      cols      = all_of(features),
      names_to  = "Feature",
      values_to = "Value"
    )
  
  ggplot(df_long, aes(x = Value)) +
    geom_point(aes(y = SoundPressure, color = "Actual"),
               size = 0.7, alpha = 0.4) +
    geom_line(aes(y = Predicted, color = "Predicted"), size = 0.8) +
    facet_wrap(~ Feature, scales = "free_x") +
    scale_color_manual(name = NULL,
                       values = c(Actual = "blue", Predicted = "red")) +
    labs(x = "Feature value", y = "SoundPressure (dB)",
         title = paste("Airfoil Predictions:", basename(model_path))) +
    theme_minimal() +
    theme(legend.position = "bottom")
}
