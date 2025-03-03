import json
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from src.evaluation.evaluation import load_data
from src.utils import *



def power_and_effect_size(y, w, alpha, alternative='larger'):
    """ y: observed outcomes
        w: intervention
        alpha: Significance level

        Returns effect size and power
    """
    y0 = y[w == 0]
    y1 = y[w == 1]
    n0 = len(y0)
    n1 = len(y1)

    # Effect size
    var1 = np.var(y1, ddof=1)
    var0 = np.var(y0, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)
    pooled_sd = np.sqrt(pooled_var)
    effect_size = (np.mean(y1) - np.mean(y0)) / pooled_sd

    # Power
    power = TTestIndPower().power(effect_size=effect_size, nobs1=n1, ratio=n0 / n1, alpha=alpha,
                                  alternative=alternative)

    return effect_size, power


def calc_eta(y, w, alpha, alternative='greater'):
    """ y: observed outcomes
        w: intervention
        alpha: Significance level

        Returns pvalue and success indication
    """
    y0 = y[w == 0]
    y1 = y[w == 1]

    # Get pvalue and success indication
    statistic, pvalue = stats.ttest_ind(y1, y0, alternative=alternative)
    eta = int(pvalue < alpha)

    return pvalue, eta


def statistical_testing(experiment_dir, output_dir, alpha=0.05, alternative='greater'):
    """ Calculate eta (success indicator) for each trial (using only RCT phase data) """
    eta_dict = {}
    trial = 0

    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        trial_key = f"trial-{trial:03d}"
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        D_t_star, D_T, D_pool = load_data(config, experiment_dir, trial=trial)

        if is_rct(config):
            # For RCT, conduct hypothesis testing on the entire trial
            w = np.array(D_T['t'])
            y = np.array(D_T['y'])
        else:
            # For adaptive trials, conduct hypothesis testing on the first phase only (RCT)
            w = np.array(D_t_star['t'])
            y = np.array(D_t_star['y'])

        pvalue, eta = calc_eta(y, w, alpha=alpha, alternative=alternative)

        eta_dict[trial_key] = eta
        trial += 1

    acquisition_function = config.get("acquisition_function")
    output_path = output_dir / f"{acquisition_function}_eta.json"
    output_path.write_text(json.dumps(eta_dict, indent=4, sort_keys=True))
