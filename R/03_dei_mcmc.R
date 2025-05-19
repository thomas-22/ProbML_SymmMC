# 03_dei_mcmc.R
# ──────────────────────────────────────────────────────────────────────────────
# Implements Deep Ensemble–Initialized MCMC (DEI–MCMC):
#  - Loads ensemble weight initializations
#  - Launches one NUTS chain per ensemble member
#  - Runs light warmup and collects posterior samples around each mode
#  - Saves chains and convergence diagnostics
