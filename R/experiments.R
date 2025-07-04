#test

rtools_bin <- "C:/rtools40/usr/bin"  
Sys.setenv(PATH = paste(rtools_bin, Sys.getenv("PATH"), sep=";"))

system("make --version")

#-------DIAGNOSTIK MODELL-----------
library(keras)

# 1) Modell laden
model <- load_model_tf("results/ensemble_airfoil/airfoil_member04_canon.keras")

# 2) Alle Gewichte extrahieren und zusammenfassen
all_weights <- get_weights(model)

summaries <- lapply(seq_along(all_weights), function(i) {
  w <- all_weights[[i]]
  data.frame(
    layer_index = i,
    shape       = paste(dim(w), collapse = "×"),
    min_val     = min(w),
    max_val     = max(w),
    mean_val    = mean(w),
    any_NA      = any(is.na(w)),
    any_Inf     = any(is.infinite(w))
  )
})

# 3) Ergebnis in einem Data Frame
summary_df <- do.call(rbind, summaries)
print(summary_df)

library(knitr)
kable(summary_df, digits = 4, caption = "Gewichts-Statistiken pro Layer")
#-------DIAGNOSTIK MODELL ENDE-----------






#-------DIAGNOSTIK DATEN-------------
library(dplyr)
library(ggplot2)
library(tidyr)
library(reshape2)

# 1) Datensatz laden
df <- readRDS("data/uci/airfoil_dataset_scaled.rds")

# 2) Grundlegende Checks
cat("Dimensionen: ", dim(df), "\n")
cat("Spaltennamen: ", names(df), "\n")
cat("NA vorhanden? ", any(is.na(df)), "\n")
cat("Inf vorhanden? ", any(is.infinite(as.matrix(df))), "\n\n")

# 3) Summary-Statistiken pro Feature
stats <- df %>% 
  summarise(
    across(everything(), list(
      min   = ~min(.x, na.rm=TRUE),
      q1    = ~quantile(.x, .25, na.rm=TRUE),
      mean  = ~mean(.x, na.rm=TRUE),
      median= ~median(.x, na.rm=TRUE),
      q3    = ~quantile(.x, .75, na.rm=TRUE),
      max   = ~max(.x, na.rm=TRUE),
      sd    = ~sd(.x, na.rm=TRUE)
    ))
  ) %>% 
  pivot_longer(everything(), names_to = c("Feature","Stat"), names_sep="_") %>% 
  pivot_wider(names_from = Stat, values_from = value)

print(stats)

# 4) Verteilung & Ausreißer (Boxplots)
df %>% pivot_longer(everything(), names_to="Feature", values_to="Value") %>%
  ggplot(aes(x=Feature, y=Value)) +
  geom_boxplot(outlier.colour="red", alpha=0.6) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1)) +
  labs(title="Boxplot: Airfoil Features & SoundPressure")

# 5) Korrelation
corr_mat <- cor(df)
# Heatmap
melted <- melt(corr_mat)
ggplot(melted, aes(Var1, Var2, fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1)) +
  labs(title="Korrelationsmatrix")

# 6) Scatter‐Pairs (stichprobenartig, wegen Größe ggf. filtern)
library(GGally)
set.seed(42)
sample_df <- df %>% sample_n(min(500, nrow(df)))
ggpairs(sample_df, columns=1:6,
        columnLabels = names(df),
        title="Pairs-Plot (Sample) aller Variablen")




df     <- readRDS("data/uci/airfoil_dataset_scaled.rds")
x_scaled <- info$x_scaled
y_vec    <- df$SoundPressure
N <- nrow(x_scaled); D <- ncol(x_scaled)









library(posterior)

# 1) List your 4 chain files
chain_files <- sprintf("results/mcmc_airfoil/airfoil_member%02d_canon_MCMC_draws.rds", 1:4)

# 2) Turn each into a draws_df with a .chain column
chain_dfs <- lapply(seq_along(chain_files), function(i) {
  d <- load_draws(chain_files[i])
  df <- as_draws_df(d)
  df$.chain     <- i
  df$.iteration <- seq_len(nrow(df))
  df
})

# 3) Stack them into one big data.frame
df_all <- do.call(rbind, chain_dfs)

# 4) Convert back to a proper draws_array
da_all <- as_draws_array(df_all)

# 5) Summarize
sum_stats <- summarize_draws(da_all)

mean(sum_stats$ess_bulk)



# 6) Pull out only the diagnostics you want
diag_df <- sum_stats[, c("variable", "rhat", "ess_bulk")]

# 1) Compute averages
avg_rhat <- mean(diag_df$rhat)
avg_ess  <- mean(diag_df$ess_bulk)

# 2) (Optional) Compute min/max to get a sense of spread
min_rhat <- min(diag_df$rhat)
max_rhat <- max(diag_df$rhat)
min_ess  <- min(diag_df$ess_bulk)
max_ess  <- max(diag_df$ess_bulk)

# 3) Print a little summary
cat(sprintf(
  "R̂  : mean = %.3f,  min = %.3f,  max = %.3f\nESS : mean = %.1f, min = %.1f, max = %.1f\n",
  avg_rhat, min_rhat, max_rhat,
  avg_ess,  min_ess,  max_ess
))



test <- load_draws("results/mcmc_airfoil/airfoil_member01_canon_MCMC_draws.rds")
test_da <- as_draws_array(test)
summarize_test <- summarize_draws(test_da)
mean(summarize_test$ess_bulk)

test <- load_draws("results/mcmc_airfoil/airfoil_member02_canon_MCMC_draws.rds")
test_da <- as_draws_array(test)
summarize_test <- summarize_draws(test_da)
mean(summarize_test$ess_bulk)

test <- load_draws("results/mcmc_airfoil/airfoil_member03_canon_MCMC_draws.rds")
test_da <- as_draws_array(test)
summarize_test <- summarize_draws(test_da)
mean(summarize_test$ess_bulk)

test <- load_draws("results/mcmc_airfoil/airfoil_member04_canon_MCMC_draws.rds")
test_da <- as_draws_array(test)
summarize_test <- summarize_draws(test_da)
mean(summarize_test$ess_bulk)



#ALL LIBRARY CALLS:
# Install and load CRAN packages if not already installed
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
library(ggplot2)

if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
library(dplyr)

if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")
library(tidyr)

if (!requireNamespace("purrr", quietly = TRUE)) install.packages("purrr")
library(purrr)

if (!requireNamespace("proxy", quietly = TRUE)) install.packages("proxy")
library(proxy)

if (!requireNamespace("posterior", quietly = TRUE)) install.packages("posterior")
library(posterior)

if (!requireNamespace("rlang", quietly = TRUE)) install.packages("rlang")
library(rlang)

# 'tools' is part of base R; no install required
library(tools)

# Install and load Keras (CRAN + Python backend) if not already installed
if (!requireNamespace("keras", quietly = TRUE)) {
  install.packages("keras")
  keras::install_keras(envname = "r-tensorflow", method = "virtualenv")
}
library(keras)

# Install and load CmdStanR (Stan interface) if not already installed
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr",
                   repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
}
library(cmdstanr)




