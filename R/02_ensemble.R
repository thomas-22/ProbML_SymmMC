# 02_ensemble.R
# ──────────────────────────────────────────────────────────────────────────────
# Deep Ensemble Training Pipeline
#  • Modular design: load any dataset, configure model, train ensemble
#  • Functions for each step, plus master `run_ensemble_pipeline()`

# 1. Install & load libraries ---------------------------------------------------
if (!requireNamespace("keras", quietly = TRUE)) {
  install.packages("keras")
  keras::install_keras(envname = "r-tensorflow", method = "virtualenv")
}
library(keras)
library(purrr)
library(ggplot2)
library(dplyr)

# 2. Configuration --------------------------------------------------------------
ensemble_size <- 4         # number of networks per dataset
epochs        <- 100        # training epochs
batch_size    <- 32        # SGD batch size
model_config  <- list(
  hidden_units = c(64, 64, 32), # hidden layers configuration
  activation   = "tanh",   # bounded activation helps MCMC
  output_units = 1
)

# 3. Modular Functions ----------------------------------------------------------

# 3.1 Load datasets from a folder (must contain .rds files with x, y)
load_datasets <- function(path = "data/synthetic") {
  files <- list.files(path, pattern = "_dataset\\.rds$", full.names = TRUE)
  data_list <- map(files, readRDS)
  names(data_list) <- tools::file_path_sans_ext(basename(files))
  return(data_list)
}

# 3.2 Build a fresh MLP model given input dimension and config
build_model <- function(input_dim, cfg) {
  model <- keras_model_sequential() %>%
    layer_dense(units = cfg$hidden_units[1], activation = cfg$activation,
                input_shape = input_dim)
  for (h in cfg$hidden_units[-1]) {
    model <- model %>% layer_dense(units = h, activation = cfg$activation)
  }
  model <- model %>%
    layer_dense(units = cfg$output_units, activation = "linear")
  model %>% compile(
    optimizer = optimizer_adam(),
    loss      = "mse",
    metrics   = list("mean_squared_error")
  )
  return(model)
}

# 3.3 Train an ensemble for one dataset
train_ensemble_for_dataset <- function(df, dataset_name, cfg, size,
                                       epochs, batch_size,
                                       save_path = "results/ensemble") {
  # Ensure save directory exists
  dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  
  x <- as.matrix(df$x)
  y <- df$y
  input_dim <- ncol(x)
  
  for (m in seq_len(size)) {
    set.seed(123 + m)
    model <- build_model(input_dim, cfg)
    model %>% fit(
      x = x, y = y,
      epochs     = epochs,
      batch_size = batch_size,
      verbose    = 0
    )
    
    # Save using the native Keras format (.keras)
    fname <- file.path(save_path,
                       sprintf("%s_member%02d.keras", dataset_name, m))
    save_model_tf(model, fname)
    message("Saved ensemble member: ", fname)
  }
}

# 3.4 Master pipeline for all datasets
run_ensemble_pipeline_synth <- function(data_path = "data/synthetic",
                                        ensemble_size = 4,
                                        epochs = 100,
                                        batch_size = 32,
                                        model_cfg = model_config,
                                        save_path = "results/ensemble_synth") {
  datasets <- load_datasets(data_path)
  walk2(datasets, names(datasets), ~ {
    message("Training ensemble for dataset: ", .y)
    train_ensemble_for_dataset(
      df           = .x,
      dataset_name = .y,
      cfg          = model_cfg,
      size         = ensemble_size,
      epochs       = epochs,
      batch_size   = batch_size,
      save_path    = save_path
    )
  })
}

# 4. Entry point ---------------------------------------------------------------
# Uncomment below to run the full pipeline
run_ensemble_pipeline_synth(ensemble_size = ensemble_size,
                            epochs = epochs,
                            batch_size = batch_size)


#OPTIONAL EVAL:
predict_x_vs_y <- function(model_path, data_path) {
  # 1. Load the model
  model <- load_model_tf(model_path)
  
  # 2. Load the data
  df <- readRDS(data_path)
  x       <- df$x
  y_true  <- df$y
  
  # 3. Predict
  y_pred <- as.numeric(model %>% predict(matrix(x, ncol = 1)))
  
  # 4. Combine into one data frame and sort by x for the line plot
  plot_df <- data.frame(
    x         = x,
    actual    = y_true,
    predicted = y_pred
  ) %>%
    arrange(x)
  
  # 5. Plot x vs actual and predicted
  ggplot(plot_df, aes(x = x)) +
    # actual data points
    geom_point(aes(y = actual, color = "Actual"),    alpha = 0.6) +
    # model predictions
    geom_line( aes(y = predicted, color = "Predicted"), size = 1) +
    # true data‐generating process y = 0.1 * x
    geom_line( aes(y = sin(2 * pi * x/2), color = "True Process"), 
               size = 1, linetype = "dashed") +
    scale_color_manual(
      name = NULL,
      values = c(
        Actual       = "blue",
        Predicted    = "red",
        "True Process" = "green"
      )
    ) +
    labs(
      title = basename(model_path),
      x     = "x",
      y     = "y value"
    ) +
    theme_minimal()
  
}

# Example usage:
predict_x_vs_y(
  model_path = "results/ensemble_synth/sin_dataset_member01.keras",
  data_path  = "data/synthetic/sin_dataset.rds"
)
