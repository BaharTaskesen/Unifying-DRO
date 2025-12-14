import numpy as np


def prepare_data(
    d=2,
    N=100,
    sparse_beta=True,
    sparsity_degree=1,
    noise_mag=0.2,
    label_noise=0,   
    beta_constrained=False
):
    X = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)

    if sparse_beta:
        beta_true = np.zeros(d)
        beta_true[:sparsity_degree] = np.random.randn(sparsity_degree)
    else:
        beta_true = np.random.randn(d)
    if beta_constrained:
        beta_true = beta_true / np.linalg.norm(beta_true, ord=2)
    scores = X @ beta_true + noise_mag * np.random.randn(N)

    y = np.sign(scores)
    y[y == 0] = 1

    # flip labels with probability label_noise
    flip = np.random.rand(N) < label_noise
    y[flip] *= -1

    return X, y


