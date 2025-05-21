# 04_symmetry.R
# ──────────────────────────────────────────────────────────────────────────────
# Performs post-hoc symmetry removal on MCMC samples:
#  - Applies sign-flip hyperplane alignment per hidden layer
#  - Executes greedy k-NN + Hungarian assignment to collapse neuron permutations
#  - Outputs canonical representative samples for unique, functionally distinct modes
