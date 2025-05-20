# 01_simulate.R
# ──────────────────────────────────────────────────────────────────────────────
# Generates and visualizes four synthetic 1D regression datasets:
#   • Defines four “true” functions (linear, sinusoid, quadratic, piecewise)
#   • Adds Gaussian noise to create observations
#   • Combines into a single data frame with a dataset identifier
#   • Overlays noise-free true functions in distinct colors
#   • Uses fixed x–axis limits (–5 to +5) and y–axis limits (–1 to +1)

library(ggplot2)
library(dplyr)
library(tidyr)

# 1. Simulation parameters
n_points <- 1250
noise_sd <- 0.25

# 2. True, noise-free functions
true_funcs <- list(
  linear    = function(x) 0.1 * x + 0,           # purely linear
  sin       = function(x) sin(2 * pi * x/2),   # sin
  quad      = function(x) (x/4)^2 - 0.5,       # quadratic
  piecewise = function(x) ifelse(x < 0, x/4, 1 - x/2.5)  # piecewise
)

# 3. Generate one data frame per true function
sim_data_list <- lapply(names(true_funcs), function(name) {
  set.seed(100 + match(name, names(true_funcs)))
  x <- runif(n_points, -5, 5)
  y <- true_funcs[[name]](x) + rnorm(n_points, sd = noise_sd)
  data.frame(x = x, y = y, dataset = name)
})
sim_data <- bind_rows(sim_data_list)

# # 4. Prepare noise-free curves on a fine grid
# grid_x <- seq(-5, 5, length.out = 500)
# true_curve_df <- bind_rows(
#   lapply(names(true_funcs), function(name) {
#     data.frame(
#       x       = grid_x,
#       y_true  = true_funcs[[name]](grid_x),
#       dataset = name
#     )
#   })
# )

# # 5. Plot: noisy data + true functions, faceted by dataset
# ggplot() +
#   geom_point(
#     data  = sim_data,
#     aes(x = x, y = y, color = dataset),
#     alpha = 0.4, size = 0.8
#   ) +
#   geom_line(
#     data = true_curve_df,
#     aes(x = x, y = y_true, color = dataset),
#     size = 1
#   ) +
#   facet_wrap(~ dataset, ncol = 2) +
#   scale_color_manual(
#     values = c(linear    = "blue",
#                sin       = "red",
#                quad      = "green",
#                piecewise = "purple")
#   ) +
#   coord_cartesian(xlim = c(-5, 5), ylim = c(-1, 1)) +
#   labs(
#     title = "Simulated Datasets with True (Noise-Free) Functions",
#     x     = "x", 
#     y     = "y"
#   ) +
#   theme_minimal() +
#   theme(
#     legend.position = "none",
#     strip.text      = element_text(face = "bold")
#   )

# # Save the datasets for further processing
# # getwd()
# for (i in seq_along(sim_data_list)) {
#   df   <- sim_data_list[[i]]
#   name <- df$dataset[1]  # read the first entry of the 'dataset' column
#   file <- file.path("data", "synthetic", paste0(name, "_dataset.rds"))
#   saveRDS(df, file = file)
#   message("Saved ", name, " to ", file)
# }
