# 02_ensemble_wrapped.R
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