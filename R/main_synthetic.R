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
  epochs        = 150,
  batch_size    = 16,
  hidden_units  = c(16, 16, 8),
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
res_synth <- cluster_canonical_models(
  canon_paths  = canon_list,
  threshold    = 0.1,
  metric       = "cosine",
  output_file  = "results/ensemble_synth/canon_cluster_eval/sin_reps_cosine.txt"
)

# Show canonicalized NN
plot_predict_x_vs_y_facets(
  model_paths = sprintf("results/ensemble_synth/sin_dataset_member%02d_canon.keras", 1:4),
  data_path   = "data/synthetic/sin_dataset.rds",
  ncol        = 2
)

#04 Run MCMC Chains that are initialized at the location of the NNs in the parameter space
# Example debug run
bnn_stan <- compile_synth_stan()

run_dei_mcmc(
  dataset_names     = "sin",
  init_file         = "results/ensemble_synth/canon_cluster_eval/sin_reps_cosine.txt",
  members_count     = 4,      # if = 0 : use every model in init_file
  warmup_steps      = 250,
  sampling_steps    = 100,
  refresh           = 1,
  adapt_delta       = 0.95,
  max_treedepth     = 15,
  subsample_frac    = 1,
  threads_per_chain = 2,
  data_path         = "data/synthetic",
  output_path       = "results/mcmc_draws",
  H1 = 16, H2 = 16, H3 = 8
)
###### NEW #######
chain_files_synth <- sprintf(
  "results/mcmc_draws/sin_chain%02d_draws.rds",
  1:4
)

chain_dfs_synth <- lapply(seq_along(chain_files_synth), function(i) {
  draws <- load_draws(chain_files_synth[i])
  df    <- as_draws_df(draws)
  df$.chain     <- i
  df$.iteration <- seq_len(nrow(df))
  df
})

df_all_synth <- do.call(rbind, chain_dfs_synth)
da_all_synth  <- as_draws_array(df_all_synth)

rhat_vals_synth    <- rhat(da_all_synth)
ess_bulk_vals_synth <- ess_bulk(da_all_synth)

print(rhat_vals_synth)      # should be ≈ 1 for every parameter
print(ess_bulk_vals_synth)  # total ESS across all 4 chains DOESNT WORK

summarize_param(df_all_synth, "sigma")
traceplot_chains(chain_files_synth,
                 "sigma")

######
# 1) Read in or bind your chains
da <- bind_draws(lapply(chain_files_synth, function(path) {
  as_draws_array(load_draws(path))
}), along = "chain")

# 2) Plot all parameters (auto-picks cols .chain, .iteration, everything else)
traceplot_all_chains(da, ncol = 2)

# 3) Or just a few, e.g. only the noise sigma and one weight
traceplot_all_chains(da,
                     pars = c("sigma", "W1[1,1]"),
                     ncol = 1
)
######



plot_credible_band_synth(
  draw_files = sprintf("results/mcmc_draws/sin_chain%02d_draws.rds", 1:4),
  nn_paths   = sprintf("results/ensemble_synth/sin_dataset_member%02d_canon.keras",1:4),
  dataset_rds_path   = "data/synthetic/sin_dataset.rds",
  H1 = 16, H2 = 16, H3 = 8,
  noise_sd = 0.2,
  level    = 0.90
)

plot_pp_mean_facets(
  draw_files = sprintf("results/mcmc_draws/sin_chain%02d_draws.rds", 1:4),
  H1 = 16, H2 = 16, H3 = 8
)

###### NEW END #######