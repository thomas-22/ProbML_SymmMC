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
  epochs           = 400,
  batch_size       = 32,
  hidden_units     = c(64, 64, 32),
  activation       = "tanh",
  output_units     = 1,
  save_path        = "results/ensemble_airfoil",
  val_split        = 0.2,
  patience         = 100
)

# 3) Plot one ensemble member vs. Frequency:
predict_airfoil_vs_y_all_features(
  model_path         = "results/ensemble_airfoil/airfoil_member01.keras",
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

#04 Run MCMC Chains that are initialized at the location of the NNs in the parameter space
# Example debug run
bnn_stan_airfoil <- compile_airfoil_stan()

run_dei_mcmc_airfoil(
  members_count      = 1,  
  init_file          = "results/ensemble_airfoil/airfoil_canon_cluster_eval/airfoil_reps_cosine.txt",
  warmup_steps       = 120,
  sampling_steps     = 50,
  refresh            = 10,
  adapt_delta        = 0.90,
  max_treedepth      = 12,
  threads_per_chain  = 12,
  dataset_scaled_rds = "data/uci/airfoil_dataset_scaled.rds",
  ensemble_path      = "results/ensemble_airfoil",
  output_path        = "results/mcmc_airfoil",
  H1                 = 64,
  H2                 = 64,
  H3                 = 32
)

#05 Inspect result of DEI MCMC
draws_mat_airfoil <- load_draws("results/mcmc_airfoil/airfoil_1chains_draws.rds")
summarize_param(draws_mat_airfoil, "sigma")
traceplot_param(draws_mat_airfoil,   "sigma")
density_param(draws_mat_airfoil,     "sigma")

# posterior mean PD
plot_posterior_predictive_mean_uciairfoil(
  draw_file         = "results/mcmc_airfoil/airfoil_1chains_draws.rds",
  dataset_rds_path  = "data/uci/airfoil_dataset.rds",
  scaler_rds_path   = "data/uci/airfoil_scaler.rds",
  H1 = 64, H2 = 64, H3 = 32,
  n.grid = 150
)

plot_posterior_predictive_samples_uciairfoil(
  draw_file         = "results/mcmc_airfoil/airfoil_1chains_draws.rds",
  dataset_rds_path  = "data/uci/airfoil_dataset.rds",
  scaler_rds_path   = "data/uci/airfoil_scaler.rds",
  H1 = 64, H2 = 64, H3 = 32,
  n.grid    = 150,
  num_draws = 30
)

plot_posterior_ensemble_and_dei(
  draw_file         = "results/mcmc_airfoil/airfoil_1chains_draws.rds",
  nn_paths_file     = "results/ensemble_airfoil/airfoil_canon_cluster_eval/airfoil_reps_cosine.txt",
  dataset_rds_path  = "data/uci/airfoil_dataset.rds",
  scaler_rds_path   = "data/uci/airfoil_scaler.rds",
  H1 = 64, H2 = 64, H3 = 32,
  n.grid    = 150,
  num_draws = 20,
  seed      = 42
)