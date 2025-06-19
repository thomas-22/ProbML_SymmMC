#Main execution file: Calls functions generated in other files.


source("R/01_simulate.R")
source("R/02_ensemble.R")
source("R/03_dei_mcmc.R")
source("R/04_inspect_draws.R")
source("R/05_symmetry.R")


#01 Generate synthetic data
simulate_data(n_points = 1250, 
              noise_sd = 0.25, 
              save_data_flag = 0, 
              plot_data_flag = 1)

#02 Train regular NN ensembles on the synthetic data
run_ensemble_pipeline(
  data_path     = "data/synthetic",
  ensemble_size = 4,
  epochs        = 100,
  batch_size    = 32,
  hidden_units  = c(64, 64, 32),
  activation    = "tanh",
  output_units  = 1,
  save_path     = "results/ensemble_synth"
)
# Show NN (illustration only)
predict_x_vs_y(
  model_path = "results/ensemble_synth/sin_dataset_member01.keras",
  data_path  = "data/synthetic/sin_dataset.rds"
)

#03 Run MCMC Chains that are initialized at the location of the NNs in the parameter space
# Example debug run
run_dei_mcmc(
  dataset_names     = "sin",
  members_count     = 1,
  warmup_steps      = 250,
  sampling_steps    = 100,
  refresh           = 5,
  adapt_delta       = 0.90,
  max_treedepth     = 12,
  subsample_frac    = 1,
  threads_per_chain = 12
)

#04 Inspect result of 03
draws_mat <- load_draws("results/mcmc_draws/sin_1chains_draws.rds")
summarize_param(draws_mat, "sigma")
traceplot_param(draws_mat,   "sigma")
density_param(draws_mat,     "sigma")

plot_posterior_predictive_mean(draws_mat, H1=64, H2=64, H3=32)

#05 Canonicalize and then cluster the ensemble models to ensure that initialized
#   chains explore functionally different modes in the posterior as opposed to
#   exploring modes that arise from symmetries.
canonicalize_model("results/ensemble_synth/sin_dataset_member01.keras")
canonicalize_model("results/ensemble_synth/sin_dataset_member02.keras")
canonicalize_model("results/ensemble_synth/sin_dataset_member03.keras")
canonicalize_model("results/ensemble_synth/sin_dataset_member04.keras")


# cluster canonicalized models and pick one representative for each symmetry group
canon_list <- sprintf("results/ensemble_synth/sin_dataset_member%02d_canon.keras",
                      1:4)
res <- cluster_canonical_models(
  canon_paths  = canon_list,
  threshold    = 0.1,
  metric       = "cosine",
  output_file  = "results/ensemble_synth/canon_cluster_eval/sin_reps_cosine.txt"
)