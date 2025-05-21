# 03_dei_mcmc.R
# ──────────────────────────────────────────────────────────────────────────────
# Deep Ensemble–Initialized MCMC Sampling Pipeline (cmdstanr + parallel chains)

# 0. Install & load libraries ---------------------------------------------------
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
}
library(cmdstanr)
# Install CmdStan if not already installed
# Check if CmdStan is installed
cmd_path <- try(cmdstanr::cmdstan_path(), silent = TRUE)
if (inherits(cmd_path, "try-error")) {
  message("CmdStan not found. Installing to default directory...")
  cmdstanr::install_cmdstan()
  cmd_path <- cmdstanr::cmdstan_path()
}
if (is.null(cmd_path) || !dir.exists(cmd_path)) {
  message("CmdStan not found. Installing to default directory...")
  cmdstanr::install_cmdstan()
}






options(mc.cores = parallel::detectCores())

library(keras)

# 1. Compile Stan model (once) ------------------------------------------------
stan_model_file <- "models/bnn_synth.stan"
message("Compiling Stan model from ", stan_model_file, " ...")
bnn_stan <- cmdstan_model(stan_model_file)
message("Stan model compiled (cmdstanr).")

# 2. Main function -------------------------------------------------------------
run_dei_mcmc <- function(
    dataset_names,           # e.g. c("linear","sin")
    members_count  = 4,      # # of ensemble members per dataset
    warmup_steps   = 200,    # HMC warmup
    sampling_steps = 800,    # post-warmup draws per chain
    refresh        = 200,    # progress interval
    adapt_delta    = 0.99,   # control for fewer divergences
    max_treedepth  = 15,     # allow deeper trees
    data_path      = "data/synthetic",
    ensemble_path  = "results/ensemble_synth",
    output_path    = "results/mcmc_draws",
    H1 = 64, H2 = 64, H3 = 32 # hidden-layer sizes
) {
  message("Running DEI-MCMC on datasets: ", paste(dataset_names, collapse = ", "))
  message(" Data path:      ", data_path)
  message(" Ensemble path:  ", ensemble_path)
  message(" Output path:    ", output_path)
  
  # locate dataset files
  pattern <- paste0("^(", paste(dataset_names, collapse="|"), ")_dataset\\.rds$")
  ds_files <- list.files(data_path, pattern=pattern, full.names=TRUE)
  if (length(ds_files)==0) stop("No dataset files found for pattern: ", pattern)
  message(" Found datasets: ", paste(basename(ds_files), collapse=", "))
  
  # locate ensemble models
  ens_files <- list.files(ensemble_path, pattern="\\.keras$", full.names=TRUE)
  if (length(ens_files)==0) stop("No ensemble models found in: ", ensemble_path)
  
  dir.create(output_path, recursive=TRUE, showWarnings=FALSE)
  
  for (ds in ds_files) {
    ds_name <- sub("_dataset\\.rds$", "", basename(ds))
    message("\n▶ Dataset: ", ds_name)
    
    # load data
    df <- readRDS(ds)
    
    #Optional: Depopulate the dataset for higher performance (debugging only)
    df <- df[sample(nrow(df), size = floor(nrow(df)/2)), ]
    
    stan_data <- list(N = nrow(df), x = df$x, y = df$y,
                      H1 = H1, H2 = H2, H3 = H3)
    
    # select ensemble members
    my_members <- grep(paste0(ds_name, "_dataset_member"), basename(ens_files), value=TRUE)
    if (length(my_members)==0) {
      warning(" No members for ", ds_name, "; skipping."); next
    }
    sel_members <- head(my_members, members_count)
    message(" Members selected: ", paste(sel_members, collapse=", "))
    sel_paths <- file.path(ensemble_path, sel_members)
    
    # build init list for all chains
    inits_list <- lapply(sel_paths, function(m_path) {
      model    <- load_model_tf(m_path)
      init_vec <- unlist(lapply(get_weights(model), as.vector))
      # index boundaries
      n_W1 <- H1; n_b1 <- H1
      n_W2 <- H2*H1; n_b2 <- H2
      n_W3 <- H3*H2; n_b3 <- H3
      n_w4 <- H3
      endW1  <- n_W1
      end_b1 <- endW1 + n_b1
      endW2  <- end_b1 + n_W2
      end_b2 <- endW2 + n_b2
      endW3  <- end_b2 + n_W3
      end_b3 <- endW3 + n_b3
      end_w4 <- end_b3 + n_w4
      b4_idx <- end_w4 + 1
      # return a named list matching Stan parameters
      list(
        W1    = init_vec[1:endW1],
        b1    = init_vec[(endW1+1):end_b1],
        W2    = matrix(init_vec[(end_b1+1):endW2], H2, H1),
        b2    = init_vec[(endW2+1):end_b2],
        W3    = matrix(init_vec[(end_b2+1):endW3], H3, H2),
        b3    = init_vec[(endW3+1):end_b3],
        w4    = init_vec[(end_b3+1):end_w4],
        b4    = init_vec[b4_idx],
        sigma = 1
      )
    })
    
    # run parallel chains
    message(" Running ", length(inits_list), " chains: warmup=", warmup_steps, ", sampling=", sampling_steps)
    fit <- bnn_stan$sample(
      data            = stan_data,
      init            = inits_list,
      chains          = length(inits_list),
      parallel_chains = length(inits_list),
      iter_warmup     = warmup_steps,
      iter_sampling   = sampling_steps,
      refresh         = refresh,
      adapt_delta     = adapt_delta,
      max_treedepth   = max_treedepth
    )
    message(" ✓ Sampling complete for ", ds_name)
    
    # save draws
    out_file <- file.path(output_path, paste0(ds_name, "_", length(inits_list), "chains_draws.rds"))
    saveRDS(fit$draws(), out_file)
    message(" ✓ Saved draws to ", out_file)
  }
  
  message("\nAll done. Posterior draws in ", output_path)
}

# --- Example Calls -------------------------------------------------------------
# Full run:
# run_dei_mcmc(
#   dataset_names  = c("linear","sin","quad","piecewise"),
#   members_count  = 4,
#   warmup_steps   = 200, sampling_steps = 800, refresh = 200
# )

# # Debug run:
run_dei_mcmc(
  dataset_names  = "sin",
  members_count  = 1,
  warmup_steps   = 50,
  sampling_steps = 250,
  refresh        = 5,
  adapt_delta    = 0.80,
  max_treedepth  = 8
)
