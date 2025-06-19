# 03_dei_mcmc.R
# -------------------------
# Deep Ensemble–Initialized MCMC Pipeline (cmdstanr + parallel chains)

# 0. Install & load libraries
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  install.packages(
    "cmdstanr",
    repos = c("https://mc-stan.org/r-packages/", getOption("repos"))
  )
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

stan_model_file <- "models/bnn_synth.stan"
  bnn_stan <- cmdstan_model(
    stan_model_file,
    force_recompile = TRUE,
    cpp_options     = list(stan_threads = TRUE)
  )
  message("CmdStan model compiled with threading enabled.")



# 2. Main function
run_dei_mcmc <- function(
    dataset_names,       # e.g. c("linear","sin",…)
    members_count = 4,   # ensemble members per dataset
    warmup_steps  = 200, # warmup iterations
    sampling_steps= 800, # draws per chain
    refresh       = 200,
    adapt_delta   = 0.99,
    max_treedepth = 15,
    subsample_frac= 1,
    threads_per_chain = 1,
    data_path     = "data/synthetic",
    ensemble_path = "results/ensemble_synth",
    output_path   = "results/mcmc_draws",
    H1 = 64, H2 = 64, H3 = 32
) {
  message("Running DEI-MCMC on: ", paste(dataset_names, collapse=", "))
  ds_files <- list.files(data_path,
                         pattern = paste0("^(",
                                          paste(dataset_names, collapse="|"),
                                          ")_dataset\\.rds$"),
                         full.names = TRUE)
  if (!length(ds_files)) stop("No datasets found.")
  ens_files <- list.files(ensemble_path,
                          pattern="\\.keras$",
                          full.names=TRUE)
  if (!length(ens_files)) stop("No .keras models found.")
  dir.create(output_path, recursive=TRUE, showWarnings=FALSE)
  
  for (ds in ds_files) {
    df <- readRDS(ds)
    if (subsample_frac < 1) {
      n_keep <- floor(nrow(df) * subsample_frac)
      df <- df[sample(nrow(df), n_keep), ]
    }
    
    # Prepare data for Stan with reduce_sum
    stan_data <- list(
      N  = nrow(df),
      x  = df$x,
      y  = df$y,
      H1 = H1,
      H2 = H2,
      H3 = H3,
      num_chunks = threads_per_chain  # Number of chunks for reduce_sum
    )
    
    # select ensemble members
    members <- grep(paste0("^", sub("_dataset\\.rds$", "", basename(ds)),
                           "_dataset_member"),
                    basename(ens_files), value=TRUE)
    sel <- head(members, members_count)
    keras_models <- map(file.path(ensemble_path, sel), load_model_tf)
    
    # Estimate scale parameters from ensemble
    estimate_scales <- function(keras_models) {
      all_weights <- map(keras_models, function(model) {
        unlist(lapply(get_weights(model), as.vector))
      })
      weight_matrix <- do.call(rbind, all_weights)
      
      # Estimate weight and bias scales
      sigma_W_est <- sd(weight_matrix)
      sigma_b_est <- sd(weight_matrix)  # Could separate if needed
      
      # Estimate noise scale from model predictions
      pred_vars <- map_dbl(keras_models, function(model) {
        preds <- predict(model, matrix(df$x, ncol = 1))
        mean((df$y - as.vector(preds))^2)
      })
      sigma_est <- sqrt(mean(pred_vars))
      
      list(
        sigma_W = max(sigma_W_est, 0.1),  # Ensure minimum scale
        sigma_b = max(sigma_b_est, 0.1),
        sigma = max(sigma_est, 0.1)
      )
    }
    
    scale_estimates <- estimate_scales(keras_models)
    
    inits_list <- map(keras_models, function(model) {
      v <- unlist(lapply(get_weights(model), as.vector))
      # extract original parameters
      n_W1 <- H1;      n_b1 <- H1
      n_W2 <- H2*H1;   n_b2 <- H2
      n_W3 <- H3*H2;   n_b3 <- H3
      n_w4 <- H3;      n_b4 <- 1
      ends <- c(
        n_W1,
        n_W1+n_b1,
        n_W1+n_b1+n_W2,
        n_W1+n_b1+n_W2+n_b2,
        n_W1+n_b1+n_W2+n_b2+n_W3,
        n_W1+n_b1+n_W2+n_b2+n_W3+n_b3,
        n_W1+n_b1+n_W2+n_b2+n_W3+n_b3+n_w4,
        n_W1+n_b1+n_W2+n_b2+n_W3+n_b3+n_w4+n_b4
      )
      W1_orig <-    v[       1:ends[1] ]
      b1_orig <-    v[(ends[1]+1):ends[2]]
      W2_orig <- matrix(v[(ends[2]+1):ends[3]], H2, H1)
      b2_orig <-    v[(ends[3]+1):ends[4]]
      W3_orig <- matrix(v[(ends[4]+1):ends[5]], H3, H2)
      b3_orig <-    v[(ends[5]+1):ends[6]]
      w4_orig <-    v[(ends[6]+1):ends[7]]
      b4_orig <-    v[ ends[7] + 1 ]
      
      # Convert to non-centered parameterization
      z_W1 <- W1_orig / scale_estimates$sigma_W
      z_b1 <- b1_orig / scale_estimates$sigma_b
      z_W2 <- W2_orig / scale_estimates$sigma_W
      z_b2 <- b2_orig / scale_estimates$sigma_b
      z_W3 <- W3_orig / scale_estimates$sigma_W
      z_b3 <- b3_orig / scale_estimates$sigma_b
      z_w4 <- w4_orig / scale_estimates$sigma_W
      z_b4 <- b4_orig / scale_estimates$sigma_b
      
      list(
        z_W1    = z_W1,
        z_b1    = z_b1,
        z_W2    = z_W2,
        z_b2    = z_b2,
        z_W3    = z_W3,
        z_b3    = z_b3,
        z_w4    = z_w4,
        z_b4    = z_b4,
        sigma_W = scale_estimates$sigma_W,
        sigma_b = scale_estimates$sigma_b,
        sigma   = scale_estimates$sigma
      )
    })
    
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
    saveRDS(fit$draws(),
            file.path(output_path,
                      sprintf("%s_%dchains_draws.rds",
                              sub("_dataset\\.rds$", "", basename(ds)),
                              length(inits_list))))
  }
}