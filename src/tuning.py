import json
import ray
from ray import tune
from ray.tune import schedulers
from ray.tune.suggest import hyperopt
from pathlib import Path

from src.models import models_utils
from src.utils import set_seeds

def tune_deep_kernel_gp(al_config, train_dataset, valid_dataset, job_dir, dim_input):
    config = al_config.copy()
    seed = config["ds_train"]["seed"]
    set_seeds(seed)

    ray.init(num_gpus=config["n_gpu"],
            dashboard_host="127.0.0.1",
            ignore_reinit_error=True,
            object_store_memory=config["object_memory_store"],
            log_to_driver=False)

    hp_space = {
        "kernel": tune.choice(["RBF", "Matern12", "Matern32", "Matern52"]),
        "num_inducing_points": tune.choice([15, 30]),
        "dim_hidden": tune.choice([50, 100, 200]),
        "depth": tune.choice([2, 3, 4]),
        "negative_slope": tune.choice([-1.0, 0.0, 0.1]),
        "dropout_rate": tune.choice([0.1, 0.2, 0.5]),
        "spectral_norm": tune.choice([0.0, 0.95, 1.5]),
        "learning_rate": tune.choice([2e-4, 5e-4, 1e-3]),
        "batch_size": tune.choice([32, 64, 100, 200])
    }

    def train_dkgp(config):
        seed = config["ds_train"]["seed"]
        set_seeds(seed)

        models_utils.TRAIN_FUNCTIONS["deep_kernel_gp"](ds_train=config["train_dataset"],
                                                       ds_valid=config["valid_dataset"],
                                                       job_dir=Path("."),
                                                       config=config,
                                                       dim_input=config["dim_input_dataset"])


    config["train_dataset"] = train_dataset
    config["valid_dataset"] = valid_dataset
    config["dim_input_dataset"] = dim_input

    algorithm = hyperopt.HyperOptSearch(hp_space, metric="mean_loss", mode="min", n_initial_points=100,
                                        random_state_seed=seed)

    scheduler = schedulers.AsyncHyperBandScheduler(grace_period=20, max_t=config.get("epochs"))

    set_seeds(seed)
    config["seed"] = seed
    analysis = tune.run(run_or_experiment=train_dkgp,
                        metric="mean_loss",
                        mode="min",
                        name="hp_dkgp",
                        resources_per_trial={
                            "cpu": config.get("cpu_per_trial"),
                            "gpu": config.get("gpu_per_trial"),
                        },
                        num_samples=50,
                        search_alg=algorithm,
                        scheduler=scheduler,
                        verbose=0,
                        local_dir=".",
                        config=config)

    best_config = analysis.best_config

    # Remove non-serializable keys
    keys_list = ['train_dataset', 'valid_dataset', 'dim_input_dataset']
    for key in keys_list:
        del best_config[key]

    hp_path = job_dir / "best_hp_config.json"
    with hp_path.open(mode="w") as hp:
        json.dump(best_config, hp)

    return best_config


TUNE_MODEL = {"deep_kernel_gp": tune_deep_kernel_gp}