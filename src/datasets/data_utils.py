"""
Code is originated from https://github.com/OATML/causal-bald/tree/fdb69553837edde0f97a5a8a647a6a24a51077a2
which is inspired by synthetic dataset presented by http://proceedings.mlr.press/v89/kallus19a/kallus19a.pdf
"""

import numpy as np


def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_ ** -1


def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_


def complete_propensity(x, u, lambda_, beta=0.75):
    nominal = nominal_propensity(x, beta=beta)
    alpha = alpha_fn(nominal, lambda_)
    beta = beta_fn(nominal, lambda_)
    return (u / alpha) + ((1 - u) / beta)


def nominal_propensity(x, beta=0.75):
    logit = beta * x + 0.5
    return (1 + np.exp(-logit)) ** -1


def f_mu(x, t, u, gamma=4.0):
    """ Note: Noise e~Ny=N(0,1) is added in the synthetic class """
    mu = ((2 * t - 1) * x
          + (2.0 * t - 1)
          - 2 * np.sin((2 * t - 2) * x)
          - (gamma * u - 2) * (1 + 0.5 * x))

    return mu
