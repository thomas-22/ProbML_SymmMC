# 03_dei_mcmc.R
# ──────────────────────────────────────────────────────────────────────────────
# Deep Ensemble–Initialized MCMC Sampling Pipeline (with debug logging)

# 0. Install & load libraries ---------------------------------------------------
if (!requireNamespace("rstan", quietly = TRUE)) install.packages("rstan", dependencies=TRUE)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

library(keras)

# 1. Compile Stan model (once) -------------------------------------------------
stan_model_file <- "models/bnn_synth.stan"
message("Compiling Stan model from ", stan_model_file, " ...")
bnn_stan <- stan_model(stan_model_file)
message("Stan model compiled.")

# 2. Main function -------------------------------------------------------------
run_dei_mcmc <- function(
    dataset_names,           # e.g. c("linear","sin")
    members_count  = 4,      # # of ensemble members per dataset
    warmup_steps   = 200,    # HMC warmup
    sampling_steps = 800,    # post-warmup draws per chain
    refresh        = 200,    # Stan progress interval
    data_path      = "data/synthetic",
    ensemble_path  = "results/ensemble_synth",
    output_path    = "results/mcmc_draws",
    H1 = 64, H2 = 64, H3 = 32 # hidden-layer sizes
) {
  message("Running DEI-MCMC on datasets: ", paste(dataset_names, collapse = ", "))
  message(" Looking in data_path:      ", data_path)
  message(" Looking in ensemble_path:  ", ensemble_path)
  message(" Output will go to:         ", output_path)
  
  # locate dataset files
  pattern <- paste0("^(", paste(dataset_names, collapse="|"), ")_dataset\\.rds$")
  ds_files <- list.files(data_path, pattern=pattern, full.names=TRUE)
  if (length(ds_files)==0) stop("No dataset files found for pattern: ", pattern)
  message(" Found dataset files: ", paste(basename(ds_files), collapse=", "))
  
  # locate all ensemble .keras files
  ens_files <- list.files(ensemble_path, pattern="\\.keras$", full.names=TRUE)
  if (length(ens_files)==0) stop("No ensemble files found in: ", ensemble_path)
  message(" Found ensemble files: ", paste(basename(ens_files), collapse=", "))
  
  dir.create(output_path, recursive=TRUE, showWarnings=FALSE)
  
  for (ds in ds_files) {
    ds_name <- sub("_dataset\\.rds$", "", basename(ds))
    message("\n▶ Sampling for dataset: ", ds_name)
    
    # load data
    df <- readRDS(ds)
    stan_data <- list(N = nrow(df), x = df$x, y = df$y,
                      H1 = H1, H2 = H2, H3 = H3)
    
    # pick members
    my_members <- grep(paste0("^", ds_name, "_dataset_member"), basename(ens_files), value=TRUE)
    if (length(my_members)==0) {
      warning("  No members found for ", ds_name, "; skipping.")
      next
    }
    message("  Members available: ", paste(my_members, collapse=", "))
    sel_members <- head(my_members, members_count)
    message("  Selecting first ", members_count, ": ", paste(sel_members, collapse=", "))
    
    # map back to full paths
    sel_paths <- file.path(ensemble_path, sel_members)
    
    # loop members
    for (m_path in sel_paths) {
      base_m <- sub("\\.keras$", "", basename(m_path))
      message("   • Member: ", base_m)
      
      # extract flat weight vector
      model    <- load_model_tf(m_path)
      w_list   <- get_weights(model)
      init_vec <- unlist(lapply(w_list, as.vector))
      message("     Extracted init_vec length: ", length(init_vec))
      
      # compute index boundaries
      n_W1 <- H1;     n_b1 <- H1
      n_W2 <- H2*H1;  n_b2 <- H2
      n_W3 <- H3*H2;  n_b3 <- H3
      n_w4 <- H3
      endW1  <- n_W1
      end_b1 <- endW1 + n_b1
      endW2  <- end_b1 + n_W2
      end_b2 <- endW2 + n_b2
      endW3  <- end_b2 + n_W3
      end_b3 <- endW3 + n_b3
      end_w4 <- end_b3 + n_w4
      b4_idx <- end_w4 + 1
      message(sprintf("     Indices: W1[1:%d], b1[%d:%d], W2[%d:%d], ..., b4[%d]", 
                      endW1, endW1+1, end_b1, end_b1+1, endW2, b4_idx))
      
      # init function for Stan
      init_fun <- function() {
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
      }
      
      # run 1 chain per member
      message("     Running Stan sampling (warmup=",warmup_steps,
              ", iter=", warmup_steps+sampling_steps, ") ...")
      fit <- sampling(
        bnn_stan, data=stan_data, init=init_fun,
        chains = 1,
        warmup = warmup_steps,
        iter   = warmup_steps + sampling_steps,
        refresh= refresh
      )
      message("     ✓ Sampling complete for ", base_m)
      
      # extract & save draws
      draws <- extract(fit)
      out_file <- file.path(output_path, paste0(base_m, "_draws.rds"))
      saveRDS(draws, out_file)
      message("    ✓ Saved draws to ", out_file)
    }
  }
  
  message("\nAll done. Posterior draws are in ", output_path)
}


# --- Example Calls -------------------------------------------------------------

# Debug run: just 'sin', 1 member, 20+40 iters, progress every 10 iters
run_dei_mcmc(
  dataset_names  = "sin",
  members_count  = 1,
  warmup_steps   = 20,
  sampling_steps = 40,
  refresh        = 10,
  ensemble_path  = "results/ensemble_synth"
)

