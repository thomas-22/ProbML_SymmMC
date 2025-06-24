# main_synthetic.R
# Main execution file for the synthetic data pipeline
# Used to calibrate methods and approaches for the UCL dataset approach


source("R/01_simulate.R")
source("R/02_ensemble.R")
source("R/03_canon.R")
source("R/04_dei_mcmc.R")
source("R/05_inspect_draws.R")



#01 Generate synthetic data
simulate_data(n_points = 1500, 
              noise_sd = 0.20, 
              save_data_flag = 0, 
              plot_data_flag = 1)

#02 Train regular NN ensembles on the synthetic data
run_ensemble_pipeline(
  data_path     = "data/synthetic",
  ensemble_size = 4,
  epochs        = 250,
  batch_size    = 32,
  hidden_units  = c(64, 64, 32),
  activation    = "tanh",
  output_units  = 1,
  save_path     = "results/ensemble_synth"
)

# Show original NN
predict_x_vs_y(
  model_path = "results/ensemble_synth/sin_dataset_member01.keras",
  data_path  = "data/synthetic/sin_dataset.rds"
)

#03 Canonicalize and then cluster the ensemble NNs to ensure that initialized
#   DEI chains explore functionally different modes in the posterior as opposed to
#   exploring modes that arise due to symmetries (neuron permutation, sign flip).
canonicalize_model("results/ensemble_synth/sin_dataset_member01.keras")
canonicalize_model("results/ensemble_synth/sin_dataset_member02.keras")
canonicalize_model("results/ensemble_synth/sin_dataset_member03.keras")
canonicalize_model("results/ensemble_synth/sin_dataset_member04.keras")

# cluster canonicalized models and pick one representative for each symmetry group
canon_list <- sprintf("results/ensemble_synth/sin_dataset_member%02d_canon.keras",
                      1:4)
cluster_canonical_models(
  canon_paths  = canon_list,
  threshold    = 0.1,
  metric       = "cosine",
  output_file  = "results/ensemble_synth/canon_cluster_eval/sin_reps_cosine.txt"
)

# Show canonicalized NN
predict_x_vs_y(
  model_path = "results/ensemble_synth/sin_dataset_member02_canon.keras",
  data_path  = "data/synthetic/sin_dataset.rds"
)

#04 Run MCMC Chains that are initialized at the location of the NNs in the parameter space
# Example debug run
bnn_stan <- compile_synth_stan()

run_dei_mcmc(
  dataset_names     = "sin",
  init_file         = "results/ensemble_synth/canon_cluster_eval/sin_reps_cosine.txt",
  members_count     = 2,      # if = 0 : use every model in init_file
  warmup_steps      = 400,
  sampling_steps    = 200,
  refresh           = 25,
  adapt_delta       = 0.95,
  max_treedepth     = 12,
  subsample_frac    = 1,
  threads_per_chain = 6,
  data_path         = "data/synthetic",
  output_path       = "results/mcmc_draws"
)

#05 Inspect result of DEI MCMC
draws_mat <- load_draws("results/mcmc_draws/sin_2chains_draws.rds")
summarize_param(draws_mat, "sigma")
traceplot_param(draws_mat,   "sigma")
density_param(draws_mat,     "sigma")

#total mean
plot_posterior_predictive_mean(draws_mat, H1=64, H2=64, H3=32)

#random samples
plot_posterior_predictive_samples(draws_mat, H1 = 64, H2 = 64, H3 = 32, num_draws = 20)

#everything together
show_sin_full_analysis(
  dataset_name = "sin",
  data_dir      = "data/synthetic",
  model_paths   = c(
    "results/ensemble_synth/sin_dataset_member01_canon.keras",
    "results/ensemble_synth/sin_dataset_member02_canon.keras"
  ),
  draw_file     = "results/mcmc_draws/sin_2chains_draws.rds",
  num_draws     = 20,
  H1 = 64, H2 = 64, H3 = 32
)

