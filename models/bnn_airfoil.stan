functions {
  // parallelized chunk likelihood
  real partial_lik(
    array[] int slice_indices,
    int start,
    int end,
    int D,                      // number of features
    matrix x,                   // N×D input matrix
    vector y,                   // N-vector of responses
    matrix W1,                  // H1×D weight matrix
    vector b1,                  // H1-unit bias
    matrix W2,                  // H2×H1 weight matrix
    vector b2,                  // H2-unit bias
    matrix W3,                  // H3×H2 weight matrix
    vector b3,                  // H3-unit bias
    vector w4,                  // H3→1 weight vector
    real   b4,                  // output bias
    real   sigma                // observation noise
  ) {
    real lp = 0;
    int slice_size = size(slice_indices);
    for (i in 1:slice_size) {
      int n = slice_indices[i];
      row_vector[D] x_n = x[n];
      vector[rows(W1)] a1 = tanh(W1 * x_n' + b1);
      vector[rows(W2)] a2 = tanh(W2 * a1   + b2);
      vector[rows(W3)] a3 = tanh(W3 * a2   + b3);
      real f_n = dot_product(w4, a3) + b4;
      lp += normal_lpdf(y[n] | f_n, sigma);
    }
    return lp;
  }
}

data {
  int<lower=1>    N;           // number of observations
  int<lower=1>    D;           // number of features (for airfoil: D=5)
  matrix[N, D]    x;           // input matrix
  vector[N]       y;           // response vector
  int<lower=1>    H1;          // hidden layer 1 size
  int<lower=1>    H2;          // hidden layer 2 size
  int<lower=1>    H3;          // hidden layer 3 size
  int<lower=1>    num_chunks;  // for reduce_sum parallelism
}

parameters {
  // non-centered parameters
  vector[H1 * D]    z_W1_flat; // flattened W1
  vector[H1]        z_b1;
  matrix[H2, H1]    z_W2;
  vector[H2]        z_b2;
  matrix[H3, H2]    z_W3;
  vector[H3]        z_b3;
  vector[H3]        z_w4;
  real              z_b4;

  // scale parameters
  real<lower=0>     sigma_W;
  real<lower=0>     sigma_B;
  real<lower=0>     sigma;
}

transformed parameters {
  // unpack non-centered into actual weights/biases
  matrix[H1, D] W1;
  {
    int idx = 1;
    for (i in 1:H1)
      for (j in 1:D) {
        W1[i, j] = sigma_W * z_W1_flat[idx];
        idx += 1;
      }
  }
  vector[H1]     b1 = sigma_B * z_b1;
  matrix[H2, H1] W2 = sigma_W * z_W2;
  vector[H2]     b2 = sigma_B * z_b2;
  matrix[H3, H2] W3 = sigma_W * z_W3;
  vector[H3]     b3 = sigma_B * z_b3;
  vector[H3]     w4 = sigma_W * z_w4;
  real           b4 = sigma_B * z_b4;
}

model {
  // Priors for non-centered parameters
  z_W1_flat ~ normal(0, 1);
  z_b1      ~ normal(0, 1);
  to_vector(z_W2) ~ normal(0, 1);
  z_b2      ~ normal(0, 1);
  to_vector(z_W3) ~ normal(0, 1);
  z_b3      ~ normal(0, 1);
  z_w4      ~ normal(0, 1);
  z_b4      ~ normal(0, 1);

  // Priors for scales
  sigma_W ~ lognormal(0, 0.5);
  sigma_B ~ lognormal(0, 0.5);
  sigma   ~ lognormal(-1, 0.5);

  // Build index array for parallelization
  array[N] int indices;
  for (n in 1:N)
    indices[n] = n;

  // Parallelized likelihood
  target += reduce_sum(
    partial_lik, indices, num_chunks,
    D, x, y,
    W1, b1,
    W2, b2,
    W3, b3,
    w4, b4,
    sigma
  );
}
