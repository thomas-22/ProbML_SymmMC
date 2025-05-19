# 02_ensemble.R
# ──────────────────────────────────────────────────────────────────────────────
# Trains a deep ensemble of M independently initialized neural networks:
#  - Builds identical MLP architectures in Keras/Torch
#  - Fits each on the same training data with different random seeds
#  - Exports trained weight sets for DEI–MCMC initialization
