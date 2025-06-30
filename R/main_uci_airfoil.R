# main_uci_airfoil.R
# Main execution file for the uci data pipeline
# Used to calibrate methods and approaches for the UC

source("R/01_simulate.R")
source("R/02_ensemble.R")
source("R/03_canon.R")
source("R/04_dei_mcmc.R")
source("R/05_inspect_draws.R")

# 1) Download & save airfoil data (run once only)
load_airfoil_data("data/uci")

prepare_and_save_airfoil(
  input_file   = "data/uci/airfoil_dataset.rds",
  output_file  = "data/uci/airfoil_dataset_scaled.rds",
  scaler_file  = "data/uci/airfoil_scaler.rds",
  scale_target = TRUE
)



# 2) Train an ensemble on Airfoil:
train_airfoil_ensemble(
  dataset_rds_path = "data/uci/airfoil_dataset_scaled.rds",
  ensemble_size    = 4,
  epochs           = 10000,
  batch_size       = 16,
  hidden_units     = c(16, 16, 8),
  activation       = "tanh",
  output_units     = 1,
  save_path        = "results/ensemble_airfoil",
  val_split        = 0.225,
  patience         = 1000
)

# 3) Plot ensemble members:

predict_airfoil_vs_y_all_features(
  model_path         = "results/ensemble_airfoil/airfoil_member01.keras",
  dataset_rds_path   = "data/uci/airfoil_dataset.rds",
  scaler_rds_path    = "data/uci/airfoil_scaler.rds"
)
predict_airfoil_vs_y_all_features(
  model_path         = "results/ensemble_airfoil/airfoil_member02.keras",
  dataset_rds_path   = "data/uci/airfoil_dataset.rds",
  scaler_rds_path    = "data/uci/airfoil_scaler.rds"
)
predict_airfoil_vs_y_all_features(
  model_path         = "results/ensemble_airfoil/airfoil_member03.keras",
  dataset_rds_path   = "data/uci/airfoil_dataset.rds",
  scaler_rds_path    = "data/uci/airfoil_scaler.rds"
)
predict_airfoil_vs_y_all_features(
  model_path         = "results/ensemble_airfoil/airfoil_member04.keras",
  dataset_rds_path   = "data/uci/airfoil_dataset.rds",
  scaler_rds_path    = "data/uci/airfoil_scaler.rds"
)


#4) Canonicalize and then cluster the ensemble NNs to ensure that initialized
#   DEI chains explore functionally different modes in the posterior as opposed to
#   exploring modes that arise due to symmetries (neuron permutation, sign flip).

canonicalize_model("results/ensemble_airfoil/airfoil_member01.keras")
canonicalize_model("results/ensemble_airfoil/airfoil_member02.keras")
canonicalize_model("results/ensemble_airfoil/airfoil_member03.keras")
canonicalize_model("results/ensemble_airfoil/airfoil_member04.keras")

# cluster canonicalized models and pick one representative for each symmetry group
canon_list_airfoil <- sprintf("results/ensemble_airfoil/airfoil_member%02d_canon.keras",
                      1:4)
res <- cluster_canonical_models(
  canon_paths  = canon_list_airfoil,
  threshold    = 0.1,
  metric       = "cosine",
  output_file  = "results/ensemble_airfoil/airfoil_canon_cluster_eval/airfoil_reps_cosine.txt"
)

# Show canonicalized NN
predict_airfoil_vs_y_all_features(
  model_path         = "results/ensemble_airfoil/airfoil_member01_canon.keras",
  dataset_rds_path   = "data/uci/airfoil_dataset.rds",
  scaler_rds_path    = "data/uci/airfoil_scaler.rds"
)
predict_airfoil_vs_y_all_features(
  model_path         = "results/ensemble_airfoil/airfoil_member02_canon.keras",
  dataset_rds_path   = "data/uci/airfoil_dataset.rds",
  scaler_rds_path    = "data/uci/airfoil_scaler.rds"
)
predict_airfoil_vs_y_all_features(
  model_path         = "results/ensemble_airfoil/airfoil_member03_canon.keras",
  dataset_rds_path   = "data/uci/airfoil_dataset.rds",
  scaler_rds_path    = "data/uci/airfoil_scaler.rds"
)
predict_airfoil_vs_y_all_features(
  model_path         = "results/ensemble_airfoil/airfoil_member04_canon.keras",
  dataset_rds_path   = "data/uci/airfoil_dataset.rds",
  scaler_rds_path    = "data/uci/airfoil_scaler.rds"
)

#04 Run MCMC Chains that are initialized at the location of the NNs in the parameter space
# Example debug run
bnn_stan_airfoil <- compile_airfoil_stan()

run_dei_mcmc_airfoil(
  members_count      = 4,  
  init_file          = "results/ensemble_airfoil/airfoil_canon_cluster_eval/airfoil_reps_cosine.txt",
  warmup_steps       = 350,
  sampling_steps     = 125,
  refresh            = 1,
  adapt_delta        = 0.95,
  max_treedepth      = 18,
  threads_per_chain  = 3,
  dataset_scaled_rds = "data/uci/airfoil_dataset_scaled.rds",
  ensemble_path      = "results/ensemble_airfoil",
  output_path        = "results/mcmc_airfoil",
  H1                 = 16,
  H2                 = 16,
  H3                 = 8
)

chain_files <- sprintf(
  "results/mcmc_airfoil/airfoil_member%02d_canon_MCMC_draws.rds",
  1:4
)

chain_dfs <- lapply(seq_along(chain_files), function(i) {
  draws <- load_draws(chain_files[i])
  df    <- as_draws_df(draws)
  df$.chain     <- i
  df$.iteration <- seq_len(nrow(df))
  df
})

df_all <- do.call(rbind, chain_dfs)
da_all  <- as_draws_array(df_all)

rhat_vals    <- rhat(da_all)
ess_bulk_vals <- ess_bulk(da_all)

print(rhat_vals)      # should be ≈ 1 for every parameter
print(ess_bulk_vals)  # total ESS across all 4 chains

summarize_param(df_all, "sigma")
traceplot_chains(sprintf("results/mcmc_airfoil/airfoil_member%02d_canon_MCMC_draws.rds", 1:4),
                 "sigma")

density_param(df_all,     "sigma")
density_param(draws2,     "sigma")

plot_pd_credible_band(
  draw_files       = sprintf("results/mcmc_airfoil/airfoil_member%02d_canon_MCMC_draws.rds", 1:4),
  dataset_rds_path = "data/uci/airfoil_dataset.rds",
  scaler_rds_path  = "data/uci/airfoil_scaler.rds",
  H1 = 16, H2 = 16, H3 = 8,
  level = 0.90,
  n_grid = 100
)


# posterior mean PD
plot_posterior_predictive_mean_uciairfoil(
  draw_file         = "results/mcmc_airfoil/airfoil_member01_canon_MCMC_draws.rds",
  dataset_rds_path  = "data/uci/airfoil_dataset.rds",
  scaler_rds_path   = "data/uci/airfoil_scaler.rds",
  H1 = 16, H2 = 16, H3 = 8,
  n.grid = 150
)

plot_posterior_predictive_samples_uciairfoil(
  draw_file         = "results/mcmc_airfoil/airfoil_member01_canon_MCMC_draws.rds",
  dataset_rds_path  = "data/uci/airfoil_dataset.rds",
  scaler_rds_path   = "data/uci/airfoil_scaler.rds",
  H1 = 16, H2 = 16, H3 = 8,
  n.grid    = 150,
  num_draws = 30
)

plot_posterior_ensemble_and_dei(
  draw_file         = "results/mcmc_airfoil/airfoil_member01_canon_MCMC_draws.rds",
  nn_paths_file     = "results/ensemble_airfoil/airfoil_canon_cluster_eval/airfoil_reps_cosine.txt",
  dataset_rds_path  = "data/uci/airfoil_dataset.rds",
  scaler_rds_path   = "data/uci/airfoil_scaler.rds",
  H1 = 16, H2 = 16, H3 = 8,
  n.grid    = 150,
  num_draws = 20,
  seed      = 42
)


plot_pd_credible_band(
  draw_files = c(
    "results/mcmc_airfoil/airfoil_member01_canon_MCMC_draws.rds",
    "results/mcmc_airfoil/airfoil_member02_canon_MCMC_draws.rds"
  ),
  dataset_rds_path = "data/uci/airfoil_dataset.rds",
  scaler_rds_path  = "data/uci/airfoil_scaler.rds",
  H1 = 16, H2 = 16, H3 = 8,
  level = 0.90,
  n_grid = 100
)