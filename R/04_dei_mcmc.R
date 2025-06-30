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
                         members_count      = 4,
                         init_file          = NULL,
                         warmup_steps       = 200,
                         sampling_steps     = 800,
                         refresh            = 200,
                         adapt_delta        = 0.99,
                         max_treedepth      = 15,
                         subsample_frac     = 1,
                         threads_per_chain  = 1,
                         data_path          = "data/synthetic",
                         ensemble_path      = "results/ensemble_synth",
                         output_path        = "results/mcmc_draws",
                         H1 = 64, H2 = 64, H3 = 32) {
  if (is.null(dataset_names))
    stop("Please specify dataset_names")
  message("Running DEI-MCMC on: ", paste(dataset_names, collapse = ", "))
  
  ds_files <- list.files(data_path,
                         pattern = paste0("^(", paste(dataset_names, collapse="|"), ")_dataset\\.rds$"),
                         full.names = TRUE)
  if (!length(ds_files))
    stop("No datasets found in ", data_path)
  dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
  
  for (ds in ds_files) {
    df <- readRDS(ds)
    if (subsample_frac < 1) {
      keep <- floor(nrow(df) * subsample_frac)
      df <- df[sample(nrow(df), keep), ]
    }
    stan_data <- list(
      N = nrow(df), x = df$x, y = df$y,
      H1 = H1, H2 = H2, H3 = H3,
      num_chunks = threads_per_chain
    )
    
    # pick init paths
    if (!is.null(init_file)) {
      inits <- readLines(init_file)
      init_paths <- if (members_count>0) head(inits, members_count) else inits
    } else {
      all_mods <- list.files(ensemble_path, pattern="\\.keras$", full.names=TRUE)
      base     <- sub("_dataset\\.rds$","",basename(ds))
      matched  <- grep(paste0("^",base,"_dataset_member"), basename(all_mods), value=TRUE)
      init_paths <- if (members_count>0) head(matched, members_count) else matched
      init_paths <- file.path(ensemble_path, init_paths)
    }
    if (!length(init_paths)) stop("No initialization models found.")
    
    keras_models <- purrr::map(init_paths, load_model_tf)
    
    # estimate scales
    estimate_scales <- function(models) {
      mats <- purrr::map(models, ~ unlist(lapply(get_weights(.x), as.vector)))
      A    <- do.call(rbind, mats)
      sW   <- max(sd(A), 0.1)
      sb   <- max(sd(A), 0.1)
      pv   <- purrr::map_dbl(models, function(m) {
        p <- m %>% predict(matrix(df$x, ncol=1))
        mean((df$y - as.vector(p))^2)
      })
      list(sigma_W = sW, sigma_b = sb, sigma = max(sqrt(mean(pv)),0.1))
    }
    scales <- estimate_scales(keras_models)
    
    # build inits
    inits_list <- purrr::map(keras_models, function(m) {
      v <- unlist(lapply(get_weights(m), as.vector))
      # splits
      nW1 <- H1; nb1 <- H1
      nW2 <- H2*H1; nb2 <- H2
      nW3 <- H3*H2; nb3 <- H3
      nw4 <- H3;   nb4 <- 1
      ends <- c(nW1, nW1+nb1, nW1+nb1+nW2, nW1+nb1+nW2+nb2,
                nW1+nb1+nW2+nb2+nW3, nW1+nb1+nW2+nb2+nW3+nb3,
                nW1+nb1+nW2+nb2+nW3+nb3+nw4,
                nW1+nb1+nW2+nb2+nW3+nb3+nw4+nb4)
      W1 <- v[1:ends[1]]
      b1 <- v[(ends[1]+1):ends[2]]
      W2 <- matrix(v[(ends[2]+1):ends[3]], H2, H1)
      b2 <- v[(ends[3]+1):ends[4]]
      W3 <- matrix(v[(ends[4]+1):ends[5]], H3, H2)
      b3 <- v[(ends[5]+1):ends[6]]
      w4 <- v[(ends[6]+1):ends[7]]
      b4 <- v[ends[7]+1]
      list(
        z_W1   = W1 / scales$sigma_W,
        z_b1   = b1 / scales$sigma_b,
        z_W2   = W2 / scales$sigma_W,
        z_b2   = b2 / scales$sigma_b,
        z_W3   = W3 / scales$sigma_W,
        z_b3   = b3 / scales$sigma_b,
        z_w4   = w4 / scales$sigma_W,
        z_b4   = b4 / scales$sigma_b,
        sigma_W = scales$sigma_W,
        sigma_b = scales$sigma_b,
        sigma   = scales$sigma
      )
    })
    
    # run
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
    
    # extract and save each chain separately
    draws_arr <- fit$draws() 
    # (this is [iter] × [chain] × [param])
    nch <- dim(draws_arr)[2]
    base <- sub("_dataset\\.rds$","", basename(ds))
    for (i in seq_len(nch)) {
      chain_mat <- draws_arr[,i, , drop=TRUE]
      fname     <- sprintf("%s_chain%02d_draws.rds", base, i)
      out       <- file.path(output_path, fname)
      saveRDS(chain_mat, out)
      message("Saved chain ", i, " to: ", out)
    }
  }
}


# Deep Ensemble–Initialized MCMC für UCI-Airfoil
run_dei_mcmc_airfoil <- function(members_count     = 4,
                                 init_file         = NULL,
                                 warmup_steps      = 200,
                                 sampling_steps    = 800,
                                 refresh           = 200,
                                 adapt_delta       = 0.99,
                                 max_treedepth     = 15,
                                 threads_per_chain = 1,
                                 
                                 # this must now point to the already-scaled dataset
                                 dataset_scaled_rds = "data/uci/airfoil_dataset_scaled.rds",
                                 
                                 ensemble_path     = "results/ensemble_airfoil",
                                 output_path       = "results/mcmc_airfoil",
                                 
                                 H1 = 64,
                                 H2 = 64,
                                 H3 = 32) {
  library(purrr)
  library(keras)
  
  # ─── A) LOAD SCALED DATA ─────────────────────────────────────────────────────
  df <- readRDS(dataset_scaled_rds)
  features  <- c("Frequency",
                 "AngleAttack",
                 "ChordLength",
                 "Velocity",
                 "SuctionThickness")
  x_scaled  <- as.matrix(df[, features])
  y_scaled  <- as.numeric(df$SoundPressure)
  N <- nrow(x_scaled)
  D <- ncol(x_scaled)
  
  stan_data <- list(
    N          = N,
    D          = D,
    x          = x_scaled,
    y          = y_scaled,
    H1         = H1,
    H2         = H2,
    H3 = H3,
    num_chunks = threads_per_chain
  )
  
  # ─── B) COLLECT INIT PATHS ───────────────────────────────────────────────────
  if (!is.null(init_file)) {
    init_lines <- readLines(init_file)
    init_paths <- head(init_lines, members_count)
  } else {
    all_canons <- list.files(ensemble_path, pattern = "_canon\\.keras$", full.names = TRUE)
    init_paths <- head(all_canons, members_count)
  }
  if (length(init_paths) < members_count)
    stop("Found only ",
         length(init_paths),
         " init models; need ",
         members_count)
  
  # ─── C) LOAD MODELS & ESTIMATE PRIOR SCALES ─────────────────────────────────
  keras_models <- map(init_paths, load_model_tf)
  
  estimate_scales <- function(models) {
    Wmat <- do.call(rbind, map(models, ~ unlist(map(
      get_weights(.x), as.vector
    ))))
    list(
      sigma_W = max(sd(Wmat), 0.1),
      sigma_B = max(sd(Wmat), 0.1),
      sigma   = 1   # since y was z-scaled
    )
  }
  scales <- estimate_scales(keras_models)
  
  # ─── D) BUILD NON-CENTERED INITS ──────────────────────────────────────────────
  inits_list <- map(keras_models, function(m) {
    v <- unlist(map(get_weights(m), as.vector))
    # exactly your original splits
    n_W1 <- H1 * D
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
    W1_flat <- v[1:ends[1]]
    b1       <- v[(ends[1] + 1):ends[2]]
    W2       <- matrix(v[(ends[2] + 1):ends[3]], H2, H1)
    b2       <- v[(ends[3] + 1):ends[4]]
    W3       <- matrix(v[(ends[4] + 1):ends[5]], H3, H2)
    b3       <- v[(ends[5] + 1):ends[6]]
    w4       <- v[(ends[6] + 1):ends[7]]
    b4       <- v[ends[7] + 1]
    
    limit <- 5   # clamp to avoid Inf
    list(
      z_W1_flat = pmin(pmax(W1_flat / scales$sigma_W, -limit), limit),
      z_b1      = pmin(pmax(b1       / scales$sigma_B, -limit), limit),
      z_W2      = pmin(pmax(W2       / scales$sigma_W, -limit), limit),
      z_b2      = pmin(pmax(b2       / scales$sigma_B, -limit), limit),
      z_W3      = pmin(pmax(W3       / scales$sigma_W, -limit), limit),
      z_b3      = pmin(pmax(b3       / scales$sigma_B, -limit), limit),
      z_w4      = pmin(pmax(w4       / scales$sigma_W, -limit), limit),
      z_b4      = pmin(pmax(b4       / scales$sigma_B, -limit), limit),
      
      sigma_W   = scales$sigma_W,
      sigma_B   = scales$sigma_B,
      sigma     = scales$sigma
    )
  })
  
  # ─── E) RUN STAN ───────────────────────────────────────────────────────────────
  fit <- bnn_stan_airfoil$sample(
    data              = stan_data,
    init              = inits_list,
    chains            = members_count,
    parallel_chains   = members_count,
    threads_per_chain = threads_per_chain,
    iter_warmup       = warmup_steps,
    iter_sampling     = sampling_steps,
    refresh           = refresh,
    adapt_delta       = adapt_delta,
    max_treedepth     = max_treedepth
  )
  # ─── F) SAVE ONE RDS PER CHAIN ───────────────────────────────────────────────
  dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
  library(tools)  # for file_path_sans_ext
  
  draws_array <- fit$draws()
  
  for (i in seq_len(members_count)) {
    # derive a file stem from the init model name
    model_name  <- file_path_sans_ext(basename(init_paths[i]))
    # extract only chain i (dimensions: iterations × 1 × variables)
    chain_draws <- draws_array[, i, , drop = FALSE]
    out_file    <- file.path(output_path, paste0(model_name, "_MCMC_draws.rds"))
    saveRDS(chain_draws, out_file)
    message("Saved Airfoil draws for chain ",
            i,
            " (",
            model_name,
            ") to: ",
            out_file)
  }
}