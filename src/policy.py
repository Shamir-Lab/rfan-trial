import numpy as np


def all_control_policy(ds):
    pi = np.zeros(ds.t.shape)
    return pi


def all_treatment_policy(ds):
    pi = np.ones(ds.t.shape)
    return pi


def model_policy(mu_0, mu_1, ds):
    """ Construct a policy using the sign of the CATE Estimator (1 if tau >= 0, otherwise 0). """
    tau_pred = (mu_1 - mu_0)
    tau_pred = tau_pred.mean(0)
    pi = np.where(tau_pred >= 0, 1, 0)
    assert pi.shape == ds.t.shape

    return pi
