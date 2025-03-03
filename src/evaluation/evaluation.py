import json
import numpy as np
from src.utils import *


# Helpers
def load_data(config, experiment_dir, trial):
    trial_dir = experiment_dir / f"trial-{trial:03d}"

    # 1. Get population of phase I only
    D_t_star = {}
    if not is_rct(config):
        switching_step = config.get("switching_step")  # last one of RCT
        phase_i_path = trial_dir / f"a-{switching_step:03d}" / "aquired.json"
        with phase_i_path.open(mode="r") as rp:
            phase_i_dict = json.load(rp)
        D_t_star = phase_i_dict["Dt"]

    # 2. Get entire trial population
    acquisition_step = config.get("max_acquisitions") - 1  # last one in trial
    entire_trial_path = trial_dir / f"a-{acquisition_step:03d}" / "aquired.json"
    with entire_trial_path.open(mode="r") as ap:
        aquired_dict = json.load(ap)
    D_T = aquired_dict["Dt"]

    # 3. Get pool population
    pool_path = trial_dir / "pool_data.json"
    with pool_path.open(mode="r") as pp:
        D_pool = json.load(pp)

    return D_t_star, D_T, D_pool


def get_objectives(output_dir, acquisition_function):
    obj_dict = {}

    pv_path = output_dir / f"{acquisition_function}_policy_value.json"
    with pv_path.open(mode="r") as pv_file:
        pv = json.load(pv_file)
    pv = np.array(list(pv.values()))

    cpv_path = output_dir / f"{acquisition_function}_control_policy_value.json"
    with cpv_path.open(mode="r") as cpv_file:
        cpv = json.load(cpv_file)
    control_pv = np.array(list(cpv.values()))

    fair_pv_path = output_dir / f"{acquisition_function}_policy_value_wc_sensitive.json"
    with fair_pv_path.open(mode="r") as fair_pv_file:
        wc_group_pv = json.load(fair_pv_file)
    wc_group_control_pv = np.array(list(wc_group_pv["subgroup_control"].values()))
    wc_group_pv = np.array(list(wc_group_pv["values"].values()))

    eta_path = output_dir / f"{acquisition_function}_eta.json"
    with eta_path.open(mode="r") as eta_file:
        eta = json.load(eta_file)
    eta = np.array(list(eta.values()))

    obj_dict["policy_value"] = pv
    obj_dict["fair_policy_value"] = wc_group_pv
    obj_dict["control_policy_value"] = control_pv
    obj_dict["wc_group_control_pv"] = wc_group_control_pv
    obj_dict["eta"] = eta

    # eta * policy value + (1-eta) * control policy value
    obj1 = (eta * pv) + ((1-eta) * control_pv)
    obj2 = (eta * wc_group_pv) + ((1-eta) * wc_group_control_pv)
    obj_dict["objective1"] = obj1
    obj_dict["objective2"] = obj2

    return obj_dict


def performance_table(output_dir, methods):
    data = {}

    for acquisition_function in methods:
        data[acquisition_function] = get_objectives(output_dir, acquisition_function)

        acc_path = output_dir / f"{acquisition_function}_accuracy.json"
        with acc_path.open(mode="r") as acc_file:
            acc = json.load(acc_file)

        pehe_path = output_dir / f"{acquisition_function}_pehe.json"
        with pehe_path.open(mode="r") as pehe_file:
            pehe = json.load(pehe_file)

        data[acquisition_function]["accuracy"] = list(acc.values())
        data[acquisition_function]["pehe"] = list(pehe.values())
        data[acquisition_function]["error_rate"] = 0.05

    return data



