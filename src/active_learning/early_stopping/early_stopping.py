import numpy as np
from scipy.stats import norm
from scipy import stats


def obrien_fleming(information_fraction, alpha=0.05):
    """ Calculate an approximation of the O'Brien-Fleming alpha spending function.

    :param information_fraction: share of the information  amount at the point of evaluation,
                                 e.g. the share of the maximum sample size
    :param alpha: type-I error rate

    :return: redistributed alpha value at the time point with the given information fraction
    """
    return (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(information_fraction))) * 2


def sequential_testing(ds_active, information_fraction, spending_function='obrien_fleming', alpha=0.05, alternative='greater'):
    """ Determine whether to stop early.

    :param ds_active: active dataset
    :param spending_function: name of the alpha spending function.
    :param estimated_sample_size: sample size to be achieved towards the end of experiment
    :param alpha: type-I error rate
    :param cap: upper bound of the adapted z-score

    :return: results of type EarlyStoppingTestStatistics
    """
    train_indices = ds_active.training_indices
    w = np.array(ds_active.dataset.t[train_indices].tolist())
    y = np.array(ds_active.dataset.y[train_indices].tolist())

    y0 = y[w == 0]
    y1 = y[w == 1]

    # alpha spending function
    if spending_function in ('obrien_fleming'):
        func = eval(spending_function)
    else:
        raise NotImplementedError
    alpha_new = func(information_fraction, alpha=alpha)

    statistic, pvalue = stats.ttest_ind(y1, y0, alternative=alternative)
    stop = int(pvalue < alpha_new)

    return stop