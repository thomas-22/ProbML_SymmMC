# 04_dei_mcmc.R
# -------------------------
# Deep Ensemble–Initialized MCMC Pipeline (cmdstanr + parallel chains)

# 0. Install & load libraries
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr",
                   repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
}
library(cmdstanr)
library(purrr)
library(keras)

# ensure CmdStan + threading is available
cmd_path <- try(cmdstanr::cmdstan_path(), silent = TRUE)
if (inherits(cmd_path, "try-error") || !dir.exists(cmd_path)) {
  message("CmdStan not found—installing...")
  cmdstanr::install_cmdstan(cores = 4, overwrite = TRUE)
}
options(mc.cores = parallel::detectCores())

# 1. Compile Stan model
compile_synth_stan <- function() {
  stan_model_file <- "models/bnn_synth.stan"
  bnn_stan <- cmdstan_model(
    stan_model_file,
    force_recompile = TRUE,
    cpp_options     = list(stan_threads = TRUE)
  )
  message("Synth CmdStan model compiled with threading enabled.")
  return(bnn_stan)
}

compile_airfoil_stan <- function() {
  stan_model_file <- "models/bnn_airfoil.stan"
  bnn_stan <- cmdstan_model(
    stan_model_file,
    force_recompile = TRUE,
    cpp_options     = list(stan_threads = TRUE)
  )
  message("Airfoil CmdStan model compiled with threading enabled.")
  return(bnn_stan)
}



# 2. Main function for synth file
run_dei_mcmc <- function(dataset_names      = NULL,
                         # e.g. c("linear","sin",…)
                         members_count      = 4,
                         # number of models to use from init_file or ensemble (0 = all)
                         init_file          = NULL,
                         # optional text file listing canonicalized model paths
                         warmup_steps       = 200,
                         # warmup iterations
                         sampling_steps     = 800,
                         # draws per chain
                         refresh            = 200,
                         adapt_delta        = 0.99,
                         max_treedepth      = 15,
                         subsample_frac     = 1,
                         threads_per_chain  = 1,
                         data_path          = "data/synthetic",
                         ensemble_path      = "results/ensemble_synth",
                         output_path        = "results/mcmc_draws",
                         H1 = 64,
                         H2 = 64,
                         H3 = 32) {
  if (is.null(dataset_names))
    stop("Please specify dataset_names")
  message("Running DEI-MCMC on: ", paste(dataset_names, collapse = ", "))
  
  # find dataset files
  ds_files <- list.files(
    data_path,
    pattern = paste0(
      "^(",
      paste(dataset_names, collapse = "|"),
      ")_dataset\\.rds$"
    ),
    full.names = TRUE
  )
  if (!length(ds_files))
    stop("No datasets found in ", data_path)
  
  dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
  
  for (ds in ds_files) {
    df <- readRDS(ds)
    if (subsample_frac < 1) {
      n_keep <- floor(nrow(df) * subsample_frac)
      df <- df[sample(nrow(df), n_keep), ]
    }
    
    stan_data <- list(
      N = nrow(df),
      x = df$x,
      y = df$y,
      H1 = H1,
      H2 = H2,
      H3 = H3,
      num_chunks = threads_per_chain
    )
    
    # Determine initialization model paths
    if (!is.null(init_file)) {
      init_list <- readLines(init_file)
      if (!length(init_list))
        stop("Initialization file is empty: ", init_file)
      init_paths <- if (members_count > 0)
        head(init_list, members_count)
      else
        init_list
    } else {
      all_mods <- list.files(ensemble_path,
                             pattern = "\\.keras$",
                             full.names = TRUE)
      base <- sub("_dataset\\.rds$", "", basename(ds))
      matched <- grep(paste0("^", base, "_dataset_member"),
                      basename(all_mods),
                      value = TRUE)
      init_paths <- if (members_count > 0)
        head(matched, members_count)
      else
        matched
      init_paths <- file.path(ensemble_path, init_paths)
    }
    if (!length(init_paths))
      stop("No initialization models found.")
    
    # Load models
    keras_models <- purrr::map(init_paths, load_model_tf)
    
    # Estimate scales
    estimate_scales <- function(models) {
      weight_list <- purrr::map(models, function(m)
        unlist(lapply(get_weights(m), as.vector)))
      mat <- do.call(rbind, weight_list)
      sigma_W <- max(sd(mat), 0.1)
      sigma_b <- max(sd(mat), 0.1)
      pred_vars <- purrr::map_dbl(models, function(m) {
        preds <- m %>% predict(matrix(df$x, ncol = 1))
        mean((df$y - as.vector(preds))^2)
      })
      sigma <- max(sqrt(mean(pred_vars)), 0.1)
      list(sigma_W = sigma_W,
           sigma_b = sigma_b,
           sigma = sigma)
    }
    scales <- estimate_scales(keras_models)
    
    # Build inits list
    inits_list <- purrr::map(keras_models, function(m) {
      v <- unlist(lapply(get_weights(m), as.vector))
      # compute parameter splits
      n_W1 <- H1
      n_b1 <- H1
      n_W2 <- H2 * H1
      n_b2 <- H2
      n_W3 <- H3 * H2
      n_b3 <- H3
      n_w4 <- H3
      n_b4 <- 1
      ends <- c(
        n_W1,
        n_W1 + n_b1,
        n_W1 + n_b1 + n_W2,
        n_W1 + n_b1 + n_W2 + n_b2,
        n_W1 + n_b1 + n_W2 + n_b2 + n_W3,
        n_W1 + n_b1 + n_W2 + n_b2 + n_W3 + n_b3,
        n_W1 + n_b1 + n_W2 + n_b2 + n_W3 + n_b3 + n_w4,
        n_W1 + n_b1 + n_W2 + n_b2 + n_W3 + n_b3 + n_w4 + n_b4
      )
      W1 <-     v[1:ends[1]]
      b1 <-     v[(ends[1] + 1):ends[2]]
      W2 <- matrix(v[(ends[2] + 1):ends[3]], H2, H1)
      b2 <-     v[(ends[3] + 1):ends[4]]
      W3 <- matrix(v[(ends[4] + 1):ends[5]], H3, H2)
      b3 <-     v[(ends[5] + 1):ends[6]]
      w4 <-     v[(ends[6] + 1):ends[7]]
      b4 <-     v[ends[7] + 1]
      # non-centered transforms
      list(
        z_W1 = W1 / scales$sigma_W,
        z_b1 = b1 / scales$sigma_b,
        z_W2 = W2 / scales$sigma_W,
        z_b2 = b2 / scales$sigma_b,
        z_W3 = W3 / scales$sigma_W,
        z_b3 = b3 / scales$sigma_b,
        z_w4 = w4 / scales$sigma_W,
        z_b4 = b4 / scales$sigma_b,
        sigma_W = scales$sigma_W,
        sigma_b = scales$sigma_b,
        sigma   = scales$sigma
      )
    })
    
    # Sample
    fit <- bnn_stan$sample(
      data              = stan_data,
      init              = inits_list,
      chains            = length(inits_list),
      parallel_chains   = length(inits_list),
      threads_per_chain = threads_per_chain,
      iter_warmup       = warmup_steps,
      iter_sampling     = sampling_steps,
      refresh           = refresh,
      adapt_delta       = adapt_delta,
      max_treedepth     = max_treedepth
    )
    
    # Save draws
    ds_name <- sub("_dataset\\.rds$", "", basename(ds))
    nchains <- length(inits_list)
    out <- file.path(output_path,
                     sprintf("%s_%dchains_draws.rds", ds_name, nchains))
    saveRDS(fit$draws(), out)
    message("Saved draws to: ", out)
  }
}

# Deep Ensemble–Initialized MCMC für UCI-Airfoil
run_dei_mcmc_airfoil <- function(
    members_count     = 4,     
    init_file         = NULL,  
    warmup_steps      = 200,
    sampling_steps    = 800,
    refresh           = 200,
    adapt_delta       = 0.99,
    max_treedepth     = 15,
    subsample_frac    = 1,
    threads_per_chain = 1,
    data_dir          = "data/uci",
    ensemble_path     = "results/ensemble_airfoil",
    output_path       = "results/mcmc_airfoil",
    H1 = 64, H2 = 64, H3 = 32
) {
  # 1) Daten laden & skalieren
  info     <- load_airfoil_scaled_inputs(data_dir)
  df       <- info$df
  x_scaled <- info$x_scaled
  y_vec    <- df$SoundPressure
  N <- nrow(x_scaled); D <- ncol(x_scaled)
  
  # 2) Stan-Datenliste
  stan_data <- list(
    N = N, D = D,
    x = x_scaled,
    y = y_vec,
    H1 = H1, H2 = H2, H3 = H3,
    num_chunks = threads_per_chain
  )
  
  # 3) Initialisierungspfade
  if (!is.null(init_file)) {
    lines      <- readLines(init_file)
    init_paths <- if (members_count>0) head(lines, members_count) else lines
  } else {
    all_canons <- list.files(ensemble_path, pattern="_canon\\.keras$", full.names=TRUE)
    init_paths <- if (members_count>0) head(all_canons, members_count) else all_canons
  }
  if (length(init_paths)==0) stop("Keine Initialisierungen gefunden")
  
  # 4) Modelle laden & Skalen schätzen
  keras_models <- purrr::map(init_paths, load_model_tf)
  scales <- estimate_scales(keras_models, x_scaled, y_vec)
  
  # 5) Inits-Liste bauen (non-centered)
  inits_list <- purrr::map(keras_models, function(m) {
    # inspect scales
    message(sprintf("Scales: sigma_W=%.3f, sigma_B=%.3f, sigma=%.3f",
                    scales$sigma_W, scales$sigma_B, scales$sigma))
    v <- unlist(lapply(get_weights(m), as.vector))
    # Parameter-Splits
    n_W1 <- H1 * D; n_b1 <- H1
    n_W2 <- H2 * H1; n_b2 <- H2
    n_W3 <- H3 * H2; n_b3 <- H3
    n_w4 <- H3;       n_b4 <- 1
    ends <- c(
      n_W1,
      n_W1 + n_b1,
      n_W1 + n_b1 + n_W2,
      n_W1 + n_b1 + n_W2 + n_b2,
      n_W1 + n_b1 + n_W2 + n_b2 + n_W3,
      n_W1 + n_b1 + n_W2 + n_b2 + n_W3 + n_b3,
      n_W1 + n_b1 + n_W2 + n_b2 + n_W3 + n_b3 + n_w4,
      n_W1 + n_b1 + n_W2 + n_b2 + n_W3 + n_b3 + n_w4 + n_b4
    )
    
    # original parameters
    W1_flat_orig <- v[1:ends[1]]
    b1_orig      <- v[(ends[1]+1):ends[2]]
    W2_orig      <- matrix(v[(ends[2]+1):ends[3]], H2, H1)
    b2_orig      <- v[(ends[3]+1):ends[4]]
    W3_orig      <- matrix(v[(ends[4]+1):ends[5]], H3, H2)
    b3_orig      <- v[(ends[5]+1):ends[6]]
    w4_orig      <- v[(ends[6]+1):ends[7]]
    b4_orig      <- v[ends[7] + 1]
    
    # non-centered transforms + cap extremes to avoid inf
    limit <- 5
    z_W1 <- pmin(pmax(W1_flat_orig / scales$sigma_W, -limit), limit)
    z_b1 <- pmin(pmax(b1_orig      / scales$sigma_B, -limit), limit)
    z_W2 <- pmin(pmax(W2_orig       / scales$sigma_W, -limit), limit)
    z_b2 <- pmin(pmax(b2_orig      / scales$sigma_B, -limit), limit)
    z_W3 <- pmin(pmax(W3_orig       / scales$sigma_W, -limit), limit)
    z_b3 <- pmin(pmax(b3_orig      / scales$sigma_B, -limit), limit)
    z_w4 <- pmin(pmax(w4_orig       / scales$sigma_W, -limit), limit)
    z_b4 <- pmin(pmax(b4_orig      / scales$sigma_B, -limit), limit)
    
    list(
      z_W1_flat = z_W1,
      z_b1      = z_b1,
      z_W2      = z_W2,
      z_b2      = z_b2,
      z_W3      = z_W3,
      z_b3      = z_b3,
      z_w4      = z_w4,
      z_b4      = z_b4,
      sigma_W   = scales$sigma_W,
      sigma_B   = scales$sigma_B,
      sigma     = scales$sigma
    )
  })
  
  # 6) Sampling starten
  fit <- bnn_stan_airfoil$sample(
    data              = stan_data,
    init              = inits_list,
    chains            = length(inits_list),
    parallel_chains   = length(inits_list),
    threads_per_chain = threads_per_chain,
    iter_warmup       = warmup_steps,
    iter_sampling     = sampling_steps,
    refresh           = refresh,
    adapt_delta       = adapt_delta,
    max_treedepth     = max_treedepth
  )
  
  # 7) Speichern
  dir.create(output_path, recursive=TRUE, showWarnings=FALSE)
  out_file <- file.path(
    output_path,
    sprintf("airfoil_%dchains_draws.rds", length(inits_list))
  )
  saveRDS(fit$draws(), out_file)
  message("Saved Airfoil draws to: ", out_file)
}

estimate_scales <- function(models, x_matrix, y_vec) {
  # 1) Flatten all weights into a big matrix
  weight_list <- lapply(models, function(m) {
    unlist(lapply(get_weights(m), as.vector))
  })
  Wmat <- do.call(rbind, weight_list)
  sigma_W <- max(stats::sd(Wmat), 0.1)
  sigma_B <- max(stats::sd(Wmat), 0.1)  # we’re using same for biases
  
  # 2) Estimate observation noise from predictive MSE
  pred_vars <- vapply(models, function(m) {
    preds <- as.numeric(m %>% predict(x_matrix))
    mean((y_vec - preds)^2)
  }, numeric(1))
  sigma <- max(sqrt(mean(pred_vars)), 0.1)
  
  list(
    sigma_W = sigma_W,
    sigma_B = sigma_B,
    sigma   = sigma
  )
}
