# 01_simulate_wrapped.R
# Simulation of synthetic regression datasets
# with optional saving and plotting

library(ggplot2)
library(dplyr)
library(tidyr)

simulate_data <- function(
    n_points       = 1250,
    noise_sd       = 0.25,
    save_data_flag = 0,
    save_dir       = "data/synthetic",
    plot_data_flag = 0
) {
  # define true functions
  true_funcs <- list(
    linear    = function(x) 0.1 * x,
    sin       = function(x) sin(2 * pi * x / 2),
    quad      = function(x) (x / 4)^2 - 0.5,
    piecewise = function(x) ifelse(x < 0, x / 4, 1 - x / 2.5)
  )
  
  # generate noisy datasets
  sim_list <- lapply(names(true_funcs), function(name) {
    set.seed(100 + match(name, names(true_funcs)))
    x_vals <- runif(n_points, -5, 5)
    y_vals <- true_funcs[[name]](x_vals) + rnorm(n_points, sd = noise_sd)
    df     <- tibble(x = x_vals, y = y_vals, dataset = name)
    
    if (as.integer(save_data_flag) == 1) {
      dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
      file <- file.path(save_dir, paste0(name, "_dataset.rds"))
      saveRDS(df, file)
      message("Saved dataset '", name, "' to: ", file)
    }
    
    df
  })
  
  sim_data <- bind_rows(sim_list)
  
  if (as.integer(plot_data_flag) == 1) {
    # prepare true-function curves
    grid_x <- seq(-5, 5, length.out = 500)
    true_df <- bind_rows(
      lapply(unique(sim_data$dataset), function(name) {
        f <- switch(name,
                    linear    = function(x) 0.1 * x,
                    sin       = function(x) sin(2 * pi * x / 2),
                    quad      = function(x) (x / 4)^2 - 0.5,
                    piecewise = function(x) ifelse(x < 0, x / 4, 1 - x / 2.5))
        tibble(x = grid_x, y = f(grid_x), dataset = name)
      })
    )
    
    # plot
    p <- ggplot() +
      geom_point(
        data = sim_data,
        aes(x = x, y = y, color = dataset),
        alpha = 0.4, size = 0.7
      ) +
      geom_line(
        data = true_df,
        aes(x = x, y = y, color = dataset),
        size = 1
      ) +
      facet_wrap(~ dataset, ncol = 2) +
      scale_color_manual(
        values = c(linear = "blue", sin = "red", quad = "green", piecewise = "purple")
      ) +
      coord_cartesian(xlim = c(-5, 5), ylim = c(-1, 1)) +
      labs(title = "Simulated Datasets with True Functions", x = "x", y = "y") +
      theme_minimal() +
      theme(legend.position = "none", strip.text = element_text(face = "bold"))
    
    print(p)
  }
  
  invisible(NULL)
}