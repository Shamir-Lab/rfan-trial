import json
import numpy as np

from src.active_learning.early_stopping.early_stopping import sequential_testing
from src.active_learning.active_learning import *
from src.models import models_utils


def active_learner_with_es(model_name, config, experiment_dir, trial, treat_prob=0.5):
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
    patient_function = config.get("patient_function")
    treatment_function = config.get("treatment_function")
    train_ratio = config.get("train_ratio")
    tune_during_trial = config.get("tune_during_trial")
    interm_frac = config.get("interm_points")
    es_path = trial_dir / "early_stopping.json"

    batch_size = step_size
    interm_points = [int(p*max_acquisitions) for p in interm_frac]
    assert interm_points[-1] == max_acquisitions, "interm points does not include 1 (full trial size)"
    switching_step = max_acquisitions # no switch
    p = [treat_prob, 1 - treat_prob]
    es_data = {}

    # Phase I - RCT with early stopping
    for i in range(max_acquisitions):
        acquisition_dir = trial_dir / f"a-{i:03d}"
        acquired_path = acquisition_dir / "aquired.json"
        if not acquired_path.exists():
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

            acquisition_dir.mkdir(parents=True, exist_ok=True)
            save_acquired_points(ds_active, acquired_path)

            # Perform intermediate analysis
            if i in interm_points:
                information_fraction = (i/max_acquisitions)
                stop_rct = sequential_testing(ds_active, information_fraction=information_fraction,
                                                             spending_function='obrien_fleming', alpha=0.05,
                                                             alternative='greater')
                if stop_rct:
                    # Initialize and train model
                    models_utils.TRAIN_FUNCTIONS[model_name](ds_train=ds_active.training_dataset,
                                                             ds_valid=ds_active.validation_dataset,
                                                             job_dir=acquisition_dir,
                                                             config=config,
                                                             dim_input=ds_active.dataset.dim_input)

                    # Save RCT info
                    switching_step = i
                    config["switching_step"] = switching_step
                    Nrct = len(ds_active.dataset.x[ds_active.training_indices])
                    es_data = {"Nrct": Nrct,
                               "switching_step": switching_step}
                    print(f"RCT Stopped (Nrct={Nrct})")

                    break

    # Phase II - Adaptive
    for i in range(switching_step+1, max_acquisitions):
        acquisition_dir = trial_dir / f"a-{i:03d}"
        acquired_path = acquisition_dir / "aquired.json"
        if not acquired_path.exists():

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

    # Write config
    config_path = trial_dir / "config.json"
    if not config_path.exists():
        with config_path.open(mode="w") as cp:
            json.dump(config, cp)

    # Write early stopping info
    if not es_path.exists():
        es_data["Ntotal"] = len(ds_active.dataset.x[ds_active.training_indices])
        with es_path.open(mode="w") as ep:
            json.dump(es_data, ep, indent=1)

    # Tune and train best model on entire trial
    model_dir = trial_dir / "final_model"
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)

        # Tune and train best model on entire trial
        _ = tune_and_retrain(model_name, config, ds_active, model_dir)


ASSIGNMENT_FUNCTIONS = {"random": assignments.random,
                        "policy": assignments.pi,
                        "mu-max": assignments.mu_max}

ACQUISITION_FUNCTIONS = {"uniform": acquisitions.uniform,
                         "mu-pi":      acquisitions.mu_pi,
                         "mu-max":  acquisitions.mu_max,
                         "sign-tau": acquisitions.sign_tau}
