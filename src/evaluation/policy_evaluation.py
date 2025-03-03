import json
import numpy as np
import torch
from torch.utils import data

from src import datasets
from src.models import models_utils
from src.policy import model_policy


def calc_tau_on_test(experiment_dir, trial):
    trial_dir = experiment_dir / f"trial-{trial:03d}"
    config_path = trial_dir / "config.json"
    with config_path.open(mode="r") as cp:
        config = json.load(cp)

    dataset_name = config.get("dataset_name")
    ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))

    x_test = ds_test.x.ravel()
    tau_true = ds_test.mu1 - ds_test.mu0

    # Sample x
    x_limit = [-3.2, 3.2]
    domain = torch.arange(x_limit[0], x_limit[1], 0.01, dtype=torch.float32).unsqueeze(-1)
    ds = data.TensorDataset(torch.cat([domain, domain], -1), domain)
    ds.dim_input = 1

    # Load last model and predict mus and predict over samples of x
    model_dir = trial_dir / "final_model"
    final_config_path = model_dir / "best_hp_config.json"
    with final_config_path.open(mode="r") as cp:
        final_config = json.load(cp)

    mu_0, mu_1 = models_utils.PREDICT_FUNCTIONS[config.get("model_name")](dataset=ds,
                                                                          job_dir=model_dir,
                                                                          config=final_config)

    tau_pred = (mu_1 - mu_0) * ds_train.y_std[0]
    domain = np.arange(x_limit[0], x_limit[1], 0.01)

    return x_test, tau_true, tau_pred, domain, x_limit


def calc_accuracy(experiment_dir, output_dir):
    """
    Calculates accuracy over test set (sign(tau_pred) / sign(tau_true)).
    """
    accuracy_dict = {}
    trial = 0

    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        trial_key = f"trial-{trial:03d}"
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")
        ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))

        tau_true = ds_test.mu1 - ds_test.mu0
        sign_tau_true = np.where(tau_true >= 0, 1, 0)

        model_dir = trial_dir / "final_model"
        final_config_path = model_dir / "best_hp_config.json"
        with final_config_path.open(mode="r") as cp:
            final_config = json.load(cp)
        mu_0, mu_1 = models_utils.PREDICT_FUNCTIONS[config.get("model_name")](dataset=ds_test,
                                                                              job_dir=model_dir,
                                                                              config=final_config)

        tau_pred = (mu_1 - mu_0) 
        tau_pred = tau_pred.mean(0)
        sign_tau_pred = np.where(tau_pred >= 0, 1, 0)

        accuracy = ((sign_tau_pred == sign_tau_true).sum() / len(sign_tau_pred)) * 100
        accuracy_dict[trial_key] = accuracy
        trial += 1

    acquisition_function = config.get("acquisition_function")

    # Store all groups
    output_path = output_dir / f"{acquisition_function}_accuracy.json"
    output_path.write_text(json.dumps(accuracy_dict, indent=4, sort_keys=True))


def fair_policy_value(config, experiment_dir, output_dir):
    """
    Calculate the policy value E[Y(pi)] given a test set and a model.

    Returns:
        float: Policy value E[Y(pi)].
    """
    policy_value = {}
    wc_pv_dict = {"values": {},
                  "subgroups": {},
                  "subgroup_control": {}}
    sensitive_covar = config.get("sensitive_covar")

    for subgroup in sensitive_covar:
        policy_value[str(subgroup)] = {}
        
    trial = 0

    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        trial_key = f"trial-{trial:03d}"
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")
        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))
        sensitive_groups = ds_test.get_sensitive_subgroups(sensitive_covar)

        # Load last model and predict mus
        model_dir = trial_dir / "final_model"
        final_config_path = model_dir / "best_hp_config.json"
        with final_config_path.open(mode="r") as cp:
            final_config = json.load(cp)

        mu_0, mu_1 = models_utils.PREDICT_FUNCTIONS[config.get("model_name")](dataset=ds_test,
                                                                              job_dir=model_dir,
                                                                              config=final_config)
        pi = model_policy(mu_0, mu_1, ds_test)

        min_pv = float("inf")
        min_group = None
        min_s = None
        for subgroup, s in sensitive_groups.items():
            N_s = s.sum()
            y_policy_value = (ds_test.y1[(s == 1) & (pi == 1)].sum() + ds_test.y0[(s == 1) & (pi == 0)].sum()) / N_s
            print(f"Sensitive group: {subgroup} ({N_s}), {y_policy_value}")
            policy_value[subgroup][trial_key] = y_policy_value
            if y_policy_value < min_pv:
                min_pv = y_policy_value
                min_group = subgroup
                min_s = s

        wc_pv_dict["values"][trial_key] = min_pv
        wc_pv_dict["subgroups"][trial_key] = min_group
        wc_pv_dict["subgroup_control"][trial_key] = float(ds_test.y0[min_s == 1].mean())
        trial += 1

    acquisition_function = config.get("acquisition_function")

    # Store all groups
    output_path = output_dir / f"{acquisition_function}_policy_value_all_sensitive_groups.json"
    output_path.write_text(json.dumps(policy_value, indent=4, sort_keys=True))

    # Store WC group
    output_path = output_dir / f"{acquisition_function}_policy_value_wc_sensitive.json"
    output_path.write_text(json.dumps(wc_pv_dict, indent=4, sort_keys=True))


def control_policy_value(experiment_dir, output_dir):
    """
    Calculate the control policy value E[Y(pi)] for failed trials.
    """
    policy_value = {}
    trial = 0

    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        trial_key = f"trial-{trial:03d}"
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")
        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))

        policy_value[trial_key] = float(ds_test.y0.mean())
        trial += 1

    print(policy_value)
    acquisition_function = config.get("acquisition_function")
    output_path = output_dir / f"{acquisition_function}_control_policy_value.json"
    output_path.write_text(json.dumps(policy_value, indent=4, sort_keys=True))


def policy_value(experiment_dir, output_dir):
    """
    Calculate the policy value E[Y(pi)] given a test set and a model.

    Returns:
        float: Policy value E[Y(pi)].
    """
    policy_value = {}
    trial = 0

    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        trial_key = f"trial-{trial:03d}"
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")
        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))

        # Load last model and predict mus
        model_dir = trial_dir / "final_model"
        final_config_path = model_dir / "best_hp_config.json"
        with final_config_path.open(mode="r") as cp:
            final_config = json.load(cp)

        mu_0, mu_1 = models_utils.PREDICT_FUNCTIONS[config.get("model_name")](dataset=ds_test,
                                                                              job_dir=model_dir,
                                                                              config=final_config)
        pi = model_policy(mu_0, mu_1, ds_test)
        y_policy_value = (ds_test.y1[pi == 1].sum() + ds_test.y0[pi == 0].sum()) / ds_test.x.shape[0]
        policy_value[trial_key] = y_policy_value
        trial += 1

    acquisition_function = config.get("acquisition_function")
    output_path = output_dir / f"{acquisition_function}_policy_value.json"
    output_path.write_text(json.dumps(policy_value, indent=4, sort_keys=True))


def pehe(experiment_dir, output_dir):
    pehe = {}
    trial = 0

    for trial_dir in sorted(experiment_dir.iterdir()):
        if "trial-" not in str(trial_dir):
            continue
        trial_key = f"trial-{trial:03d}"
        config_path = trial_dir / "config.json"
        with config_path.open(mode="r") as cp:
            config = json.load(cp)

        dataset_name = config.get("dataset_name")
        ds_test = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))

        # Load last model and predict mus
        model_dir = trial_dir / "final_model"
        final_config_path = model_dir / "best_hp_config.json"
        with final_config_path.open(mode="r") as cp:
            final_config = json.load(cp)

        mu_0, mu_1 = models_utils.PREDICT_FUNCTIONS[config.get("model_name")](dataset=ds_test,
                                                                              job_dir=model_dir,
                                                                              config=final_config)
        tau_pred = (mu_1 - mu_0) 
        tau_true = ds_test.mu1 - ds_test.mu0
        pehe[trial_key] = float(rmse_fn(tau_pred.mean(0), tau_true))
        trial += 1

    acquisition_function = config.get("acquisition_function")
    pehe_path = output_dir / f"{acquisition_function}_pehe.json"
    pehe_path.write_text(json.dumps(pehe, indent=4, sort_keys=True))


def rmse_fn(y_pred, y):
    return np.sqrt(np.mean(np.square(y_pred - y)))