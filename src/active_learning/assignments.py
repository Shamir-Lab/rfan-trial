# Treatment Assignment
import numpy as np

from src.policy import model_policy
from src.active_learning.acquisitions import mu


def random(mu_0, mu_1, ds, p, seed):
    """ Returns random treatment arms """
    np.random.seed(seed+1) # seed should be different than the one used in uniform() to avoid unintended correlations
    return np.random.choice([0, 1], size=ds.dataset.t.shape, p=p)


def pi(mu_0, mu_1, ds, p, seed):
    """ Returns treatment assignment by model policy """
    return model_policy(mu_0, mu_1, ds.dataset)


def mu_max(mu_0, mu_1, ds, p, seed):
    """ Returns
        scores: patient-wise maximum mu scores, considering both treatment arms """
    # scores are independent of treatment recommended by current policy
    scores_control = mu(mu_0=mu_0, mu_1=mu_1, t=0, seed=seed)
    scores_treatment = mu(mu_0=mu_0, mu_1=mu_1, t=1, seed=seed)
    scores = np.maximum(scores_control, scores_treatment)
    uncertain_treatments = np.argmax(np.vstack((scores_control, scores_treatment)), axis=0)

    return scores, uncertain_treatments
