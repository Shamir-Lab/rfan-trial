# Patient Acquisition
import numpy as np
from src.policy import model_policy


def uniform(mu_0, mu_1, t, ds, seed):
    """ Returns random acquisition scores between [0.0, 1.0) """
    np.random.seed(seed)
    return np.random.random(t.shape)


def mu(mu_0, mu_1, t, seed):
    # I(Yt | x, t, D_train) ≈ Var(µ(x,t))
    # For each datapoint x and treatment t, it calculates Var(µ(x,t)) over the drwan 1000 posterior samples
    # Returns numpy array of log(Var(µ(x,t))) of size `t.shape`
    return np.log((t * mu_1.var(0) + (1 - t) * mu_0.var(0)))



def mu_pi(mu_0, mu_1, t, ds, seed):
    # I(Yt | x, t, D_train) ≈ Var(µ(x,t)) given current policy
    t_pi = model_policy(mu_0, mu_1, ds.dataset)
    return mu(mu_0, mu_1, t_pi, seed)


def mu_max(mu_0, mu_1, t, ds, seed):
    """ Returns
        scores: patient-wise maximum mu scores, considering both treatment arms """

    # scores are independent of treatment recommended by current policy
    scores_control = mu(mu_0=mu_0, mu_1=mu_1, t=0)
    scores_treatment = mu(mu_0=mu_0, mu_1=mu_1, t=1)
    scores = np.maximum(scores_control, scores_treatment)

    return scores


def sign_tau(mu_0, mu_1, t, ds, seed):
    """ Var(sign(µ(x, 1) − µ(x, 0))) """
    tau_pred = (mu_1 - mu_0)
    sign_tau = np.where(tau_pred >= 0, 1, 0)

    return sign_tau.var(0)
