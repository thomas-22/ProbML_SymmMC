data {
  int<lower=1> N;           // number of observations
  vector[N]    x;           // input
  vector[N]    y;           // output
  int<lower=1> H1;          // hidden_units[1]
  int<lower=1> H2;          // hidden_units[2]
  int<lower=1> H3;          // hidden_units[3]
}
parameters {
  vector[H1]     W1;        // weights from input to first hidden layer
  vector[H1]     b1;
  matrix[H2,H1]  W2;
  vector[H2]     b2;
  matrix[H3,H2]  W3;
  vector[H3]     b3;
  vector[H3]     w4;        // weights from third hidden layer to output
  real           b4;        // output bias
  real<lower=0>  sigma;
}
transformed parameters {
  vector[N]      f;
  matrix[H1, N]  a1;
  matrix[H2, N]  a2;
  matrix[H3, N]  a3;

  for (n in 1:N) {
    // first hidden layer
    a1[, n] = tanh(W1 * x[n] + b1);
    // second hidden layer
    a2[, n] = tanh(W2 * a1[, n] + b2);
    // third hidden layer
    a3[, n] = tanh(W3 * a2[, n] + b3);
    // output layer
    f[n]    = dot_product(w4, a3[, n]) + b4;
  }
}
model {
  // priors
  W1 ~ normal(0, 1);
  b1 ~ normal(0, 1);
  to_vector(W2) ~ normal(0, 1);
  b2 ~ normal(0, 1);
  to_vector(W3) ~ normal(0, 1);
  b3 ~ normal(0, 1);
  w4 ~ normal(0, 1);
  b4 ~ normal(0, 1);
  sigma ~ normal(0, 1);

  // likelihood
  y ~ normal(f, sigma);
}
