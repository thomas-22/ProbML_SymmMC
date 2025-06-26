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