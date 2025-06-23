functions {
  real partial_lik(
    array[] int slice_indices,
    int start,
    int end,
    data array[] real x,
    data array[] real y,
    vector W1,
    vector b1,
    matrix W2,
    vector b2,
    matrix W3,
    vector b3,
    vector w4,
    real b4,
    real sigma
  ) {
    real lp = 0;
    int slice_size = size(slice_indices);
    for (i in 1:slice_size) {
      int n = slice_indices[i];
      real x_n = x[n];
      real y_n = y[n];
      vector[rows(W1)] a1 = tanh(W1 * x_n + b1);
      vector[rows(W2)] a2 = tanh(W2 * a1 + b2);
      vector[rows(W3)] a3 = tanh(W3 * a2 + b3);
      real f_n = dot_product(w4, a3) + b4;
      lp += normal_lpdf(y_n | f_n, sigma);
    }
    return lp;
  }
}
data {
  int<lower=1>    N;           // number of observations
  array[N] real   x;           // input
  array[N] real   y;           // output
  int<lower=1>    H1;          // hidden_units[1]
  int<lower=1>    H2;          // hidden_units[2]
  int<lower=1>    H3;          // hidden_units[3]
  int<lower=1>    num_chunks;  // for reduce_sum parallelism
}
parameters {
  vector[H1]        z_W1;
  vector[H1]        z_b1;
  matrix[H2, H1]    z_W2;
  vector[H2]        z_b2;
  matrix[H3, H2]    z_W3;
  vector[H3]        z_b3;
  vector[H3]        z_w4;
  real              z_b4;
  real<lower=0>     sigma_W;
  real<lower=0>     sigma_b;
  real<lower=0>     sigma;
}
transformed parameters {
  // unpack non-centered parameters
  vector[H1]        W1 = sigma_W * z_W1;
  vector[H1]        b1 = sigma_b * z_b1;
  matrix[H2, H1]    W2 = sigma_W * z_W2;
  vector[H2]        b2 = sigma_b * z_b2;
  matrix[H3, H2]    W3 = sigma_W * z_W3;
  vector[H3]        b3 = sigma_b * z_b3;
  vector[H3]        w4 = sigma_W * z_w4;
  real              b4 = sigma_b * z_b4;
}
model {
  // Create index array for reduce_sum
  array[N] int indices;
  for (n in 1:N) {
    indices[n] = n;
  }
  
  // non-centered priors
  z_W1 ~ normal(0, 1);
  z_b1 ~ normal(0, 1);
  to_vector(z_W2) ~ normal(0, 1);
  z_b2 ~ normal(0, 1);
  to_vector(z_W3) ~ normal(0, 1);
  z_b3 ~ normal(0, 1);
  z_w4 ~ normal(0, 1);
  z_b4 ~ normal(0, 1);
  
  // Better priors for scale parameters
  sigma_W ~ lognormal(0, 0.5);  // log-normal with median 1, reasonable spread
  sigma_b ~ lognormal(0, 0.5);  // log-normal with median 1, reasonable spread
  sigma   ~ lognormal(-1, 0.5); // log-normal with median ~0.37, appropriate for noise
  
  // parallelized likelihood
  target += reduce_sum(partial_lik, indices, num_chunks,
                       x, y,
                       W1, b1,
                       W2, b2,
                       W3, b3,
                       w4, b4,
                       sigma);
}
