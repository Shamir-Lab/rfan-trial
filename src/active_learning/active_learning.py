import json
import random
import torch
import numpy as np

from src import datasets, tuning
from src.active_learning import acquisitions, assignments
from src.models import models_utils
from src.evaluation.statistical_testing import calc_eta
from src.utils import *


def acquisition_and_assignment_scores(patient_function, treatment_function, mu_0, mu_1, ds, p, seed):
    """ Args:
            patient_selection: 'uniform', 'mu-pi', 'mu-pi', 'sign-tau'
            treatment_assigment: 'random', 'policy', 'mu-max', 'sign-tau'
        Returns: patient-wise acquisition_scores and assignments
    """
    if treatment_function == "mu-max": # Returns both scores and treatment together
        return ASSIGNMENT_FUNCTIONS[treatment_function](mu_0, mu_1, ds, p, seed)

    assignments = ASSIGNMENT_FUNCTIONS[treatment_function](mu_0, mu_1, ds, p, seed)
    acquisition_scores = ACQUISITION_FUNCTIONS[patient_function](mu_0, mu_1, assignments, ds, seed)

    return acquisition_scores, assignments


def valid_rct(acquisition_scores, treatment_assignments, ds_active, p):
    # While all patients were assigned into the same treatment arm, re-randomised
    # We use this helper only in cases where switching step=0 (for illustrative purposes)
    while np.all(treatment_assignments == treatment_assignments[0]):
        print("All patients were assigned into the same treatment arm, re-randomised")
        acquisition_scores, treatment_assignments = acquisition_and_assignment_scores(patient_function="uniform",
                                                                                      treatment_function="random",
                                                                                      mu_0=None, mu_1=None,
                                                                                      ds=ds_active, p=p)

    return acquisition_scores, treatment_assignments


def save_acquired_points(ds_active, acquired_path):
    train_indices = ds_active.training_indices
    val_indices = ds_active.validation_indices
    with acquired_path.open(mode="w") as ap:
        json.dump(
            {"train_indices": [int(a) for a in train_indices],
             "val_indices": [int(a) for a in val_indices],
             "Dt": {"x": ds_active.dataset.x[train_indices].tolist(),
                    "t": ds_active.dataset.t[train_indices].tolist(),
                    "y": ds_active.dataset.y[train_indices].tolist()},
             "Dval": {"x": ds_active.dataset.x[val_indices].tolist(),
                      "t": ds_active.dataset.t[val_indices].tolist(),
                      "y": ds_active.dataset.y[val_indices].tolist()}
             },
            ap,
        )


def tune_and_retrain(model_name, config, ds_active, model_dir):
    # Hyperparameter search
    best_config = tuning.TUNE_MODEL[model_name](al_config=config,
                                                train_dataset=ds_active.training_dataset,
                                                valid_dataset=ds_active.validation_dataset,
                                                job_dir=model_dir,
                                                dim_input=ds_active.dataset.dim_input)

    # Train best model (overwrite if exists)
    models_utils.TRAIN_FUNCTIONS[model_name](ds_train=ds_active.training_dataset,
                                             ds_valid=ds_active.validation_dataset,
                                             job_dir=model_dir,
                                             config=best_config,
                                             dim_input=ds_active.dataset.dim_input,
                                             overwrite_model=True)

    return best_config


def run_rct(model_name, config, experiment_dir, trial, treat_prob=0.5):
    # Set dataset seeds
    dataset_name = config.get("dataset_name")
    config["ds_train"]["seed"] = trial
    config["ds_test"]["seed"] = trial + 1 if dataset_name == "synthetic" else trial

    # Get datasets
    ds_active = datasets.ActiveLearningDataset(datasets.DATASETS.get(dataset_name)(**config.get("ds_train")))

    # Set the trial dir
    experiment_dir = models_utils.DIRECTORIES[model_name](base_dir=experiment_dir, config=config)
    trial_dir = experiment_dir / f"trial-{trial:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Write config for downstream use
    config_path = trial_dir / "config.json"
    with config_path.open(mode="w") as cp:
        json.dump(config, cp)

    # Store pool data
    pool_path = trial_dir / "pool_data.json"
    data = {"x": ds_active.dataset.x.tolist(),
            "y0": ds_active.dataset.y0.tolist(),
            "y1": ds_active.dataset.y1.tolist()}
    with pool_path.open(mode="w") as pp:
        json.dump(data, pp, indent=1)

    # Active learning loop
    step_size = config.get("step_size")
    max_acquisitions = config.get("max_acquisitions")
    train_ratio = config.get("train_ratio")
    p = [treat_prob, 1 - treat_prob]

    acquisition_dir = trial_dir / f"a-{max_acquisitions-1:03d}"
    acquired_path = acquisition_dir / "aquired.json"
    if not acquired_path.exists():
        # Phase I - RCT
        batch_size = max_acquisitions * step_size
        acquisition_scores, treatment_assignments = acquisition_and_assignment_scores(patient_function="uniform",
                                                                                      treatment_function="random",
                                                                                      mu_0=None, mu_1=None,
                                                                                      ds=ds_active, p=p,
                                                                                      seed=config["ds_train"]["seed"])

        # Filter only scores remain available
        acquisition_scores = acquisition_scores[ds_active.pool_dataset.indices]

        # Acquire data points by taking the points with the maximal acquisition scores
        idx = np.argsort(acquisition_scores)[-batch_size:]
        ds_active.acquire(idx, treatment_assignments, train_ratio=train_ratio)

        # Initialize and train model
        models_utils.TRAIN_FUNCTIONS[model_name](
            ds_train=ds_active.training_dataset,
            ds_valid=ds_active.validation_dataset,
            job_dir=acquisition_dir,
            config=config,
            dim_input=ds_active.dataset.dim_input,
        )

        save_acquired_points(ds_active, acquired_path)

    model_dir = trial_dir / "final_model"
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)

        # Tune and train best model on entire trial
        _ = tune_and_retrain(model_name, config, ds_active, model_dir)


def active_learner(model_name, config, experiment_dir, trial, treat_prob=0.5):
    # Set dataset seeds
    dataset_name = config.get("dataset_name")
    config["ds_train"]["seed"] = trial
    config["ds_test"]["seed"] = trial + 1 if dataset_name == "synthetic" else trial

    # Get datasets
    ds_active = datasets.ActiveLearningDataset(datasets.DATASETS.get(dataset_name)(**config.get("ds_train")))

    # Set the trial dir
    experiment_dir = models_utils.DIRECTORIES[model_name](base_dir=experiment_dir, config=config)
    trial_dir = experiment_dir / f"trial-{trial:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Write config for downstream use
    config_path = trial_dir / "config.json"
    with config_path.open(mode="w") as cp:
        json.dump(config, cp)

    # Store pool data
    pool_path = trial_dir / "pool_data.json"
    data = {"x": ds_active.dataset.x.tolist(),
            "y0": ds_active.dataset.y0.tolist(),
            "y1": ds_active.dataset.y1.tolist()}
    with pool_path.open(mode="w") as pp:
        json.dump(data, pp, indent=1)

    # Active learning loop
    step_size = config.get("step_size")
    max_acquisitions = config.get("max_acquisitions")
    switching_step = config.get("switching_step")
    patient_function = config.get("patient_function")
    treatment_function = config.get("treatment_function")
    train_ratio = config.get("train_ratio")
    tune_during_trial = config.get("tune_during_trial")

    p = [treat_prob, 1 - treat_prob]

    for i in range(switching_step, max_acquisitions):
        acquisition_dir = trial_dir / f"a-{i:03d}"
        acquired_path = acquisition_dir / "aquired.json"
        if not acquired_path.exists():
            # Phase I - RCT
            if i <= switching_step:
                batch_size = (switching_step + 1) * step_size
                acquisition_scores, treatment_assignments = acquisition_and_assignment_scores(patient_function="uniform",
                                                                                              treatment_function="random",
                                                                                              mu_0=None, mu_1=None,
                                                                                              ds=ds_active, p=p,
                                                                                              seed=config["ds_train"]["seed"])
                if (i == 0) and (switching_step == 0): # Small illustrative RCT - requires validation
                    acquisition_scores, treatment_assignments = valid_rct(acquisition_scores, treatment_assignments,
                                                                          ds_active, p)
            # Phase II - Adaptive
            else:
                batch_size = step_size

                # If first adaptive step - tune model
                if tune_during_trial:
                    # Overwrite last model and config with tuned hp, according to the entire RCT phase
                    config = tune_and_retrain(model_name, config, ds_active, model_dir=trial_dir / f"a-{i - 1:03d}")

                # Load last model (i-1) and predict on pool set (N=10K)
                mu_0, mu_1 = models_utils.PREDICT_FUNCTIONS[model_name](dataset=ds_active.dataset,
                                                                        job_dir=trial_dir / f"a-{i - 1:03d}",
                                                                        config=config)

                # Get acquisition scores and treatment assignment
                acquisition_scores, treatment_assignments = acquisition_and_assignment_scores(patient_function=patient_function,
                                                                                              treatment_function=treatment_function,
                                                                                              mu_0=mu_0, mu_1=mu_1,
                                                                                              ds=ds_active, p=p,
                                                                                              seed=config["ds_train"]["seed"])

            # Filter only scores remain available
            acquisition_scores = acquisition_scores[ds_active.pool_dataset.indices]

            # Acquire data points by taking the points with the maximal acquisition scores
            idx = np.argsort(acquisition_scores)[-batch_size:]
            ds_active.acquire(idx, treatment_assignments, train_ratio=train_ratio)

            # Initialize and train model
            models_utils.TRAIN_FUNCTIONS[model_name](
                ds_train=ds_active.training_dataset,
                ds_valid=ds_active.validation_dataset,
                job_dir=acquisition_dir,
                config=config,
                dim_input=ds_active.dataset.dim_input,
            )

            save_acquired_points(ds_active, acquired_path)

    model_dir = trial_dir / "final_model"
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)

        # Tune and train best model on entire trial
        _ = tune_and_retrain(model_name, config, ds_active, model_dir)
        
        
def draw_multiple_trials(N_seeds, config, alpha, alternative, custom_rct_size=None, treat_prob=0.5):
    """ Conduct statistical testing N_seeds time over simulated randomized data """
    eta_ls = []
    dataset_name = config.get("dataset_name")
    switching_step = config.get("switching_step")
    step_size = config.get("step_size")
    max_acquisitions = config.get("max_acquisitions")
    train_ratio = config.get("train_ratio")
    
    if is_rct(config):
        if custom_rct_size:
            batch_size = custom_rct_size
        else:        
            batch_size = max_acquisitions * step_size
    else:
        if custom_rct_size:
            batch_size = custom_rct_size / 2
        else:
            batch_size = (switching_step + 1) * step_size

    for trial in range(N_seeds):
        # Set dataset seeds
        config["ds_train"]["seed"] = trial
        config["ds_test"]["seed"] = trial + 1 if dataset_name == "synthetic" else trial

        ds_active = datasets.ActiveLearningDataset(datasets.DATASETS.get(dataset_name)(**config.get("ds_train")))
        p = [treat_prob, 1 - treat_prob]

        acquisition_scores, treatment_assignments = acquisition_and_assignment_scores(patient_function="uniform",
                                                                                      treatment_function="random",
                                                                                      mu_0=None, mu_1=None,
                                                                                      ds=ds_active, p=p,
                                                                                      seed=config["ds_train"]["seed"])

        # Filter only scores remain available
        acquisition_scores = acquisition_scores[ds_active.pool_dataset.indices]

        # Acquire data points by taking the points with the maximal acquisition scores
        idx = np.argsort(acquisition_scores)[-batch_size:]
        ds_active.acquire(idx, treatment_assignments, train_ratio=train_ratio)

        train_indices = ds_active.training_indices
        w = np.array(ds_active.dataset.t[train_indices].tolist())
        y = np.array(ds_active.dataset.y[train_indices].tolist())

        pvalue, eta = calc_eta(y, w, alpha=alpha, alternative=alternative)
        eta_ls.append(eta)

    return eta_ls


ASSIGNMENT_FUNCTIONS = {"random": assignments.random,
                        "policy": assignments.pi,
                        "mu-max": assignments.mu_max}

ACQUISITION_FUNCTIONS = {"uniform": acquisitions.uniform,
                         "mu-pi":      acquisitions.mu_pi,
                         "mu-max":  acquisitions.mu_max,
                         "sign-tau": acquisitions.sign_tau}
