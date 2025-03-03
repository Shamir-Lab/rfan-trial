# Set configurations of dataset, models and experiments
from pathlib import Path


def active_learning_config(context, job_dir, num_trials, step_size, max_acquisitions, switching_step, patient_function,
                           treatment_function, acquisition_function, tune_during_trial, gpu_per_trial, cpu_per_trial,
                           object_memory_store):
    job_dir = (
        Path(job_dir)
        / "al"
        / f"ss-{step_size}_ma-{max_acquisitions}_af-{acquisition_function}_t-{tune_during_trial}"
    )
    context.update(
        {
            "job_dir": str(job_dir),
            "num_trials": num_trials,
            "step_size": step_size,
            "max_acquisitions": max_acquisitions,
            "switching_step": switching_step,
            "patient_function": patient_function,
            "treatment_function": treatment_function,
            "acquisition_function": acquisition_function,
            "tune_during_trial": tune_during_trial,
            "gpu_per_trial": gpu_per_trial,
            "cpu_per_trial": cpu_per_trial,
            "object_memory_store": object_memory_store,
        }
    )   


def active_learning_with_es_config(context, job_dir, num_trials, step_size, max_acquisitions, switching_step, patient_function,
                                    treatment_function, acquisition_function, tune_during_trial, interm_points,
                                    gpu_per_trial, cpu_per_trial, object_memory_store):

    active_learning_config(context, job_dir, num_trials, step_size, max_acquisitions, switching_step, patient_function,
                           treatment_function, acquisition_function, tune_during_trial, gpu_per_trial, cpu_per_trial,
                           object_memory_store)
    context.update(
        {
            "interm_points": interm_points,
        }
    )


def synthetic_config(context, num_examples=10000, beta=2.0, bimodal=False, sigma=1.0):
    dataset_name = "synthetic"
    job_dir = context.get("job_dir")
    if job_dir is not None:
        experiment_dir = (
            Path(job_dir)
            / f"be-{beta:.02f}_si-{sigma:.02f}"
        )
    else:
        experiment_dir = None
    context.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "sensitive_covar": [(-5, -1.2), (1.3, 5)],
            "test_alternative": "greater",
            "ds_train": {
                "num_examples": num_examples,
                "mode": "mu",
                "beta": beta,
                "bimodal": bimodal,
                "sigma_y": sigma,
            },
            "ds_test": {
                "num_examples": min(num_examples, 2000),
                "mode": "mu",
                "beta": beta,
                "bimodal": bimodal,
                "sigma_y": sigma,
            },
        }
    )
    

def iwpc_config(context):
    dataset_name = "iwpc"
    data_path = "./datasets/iwpc/parsed/iwpc_parsed.csv"
    job_dir = context.get("job_dir")
    if job_dir is not None:
        experiment_dir = Path(job_dir)
    else:
        experiment_dir = None
    context.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "sensitive_covar": ["male_white", "male_black", "male_asian", 
                                "female_white", "female_black", "female_asian"],
            "test_alternative": "two-sided",
            "ds_train": {
                "data_path": data_path,
                "split": "train",
                "seed": context.get("seed"),
            },
            "ds_valid": {
                "data_path": data_path,
                "split": "valid",
                "seed": context.get("seed"),
            },
            "ds_test": {
                "data_path": data_path,
                "split": "test",
                "seed": context.get("seed"),
            },
        }
    )

def covid_config(context):
    dataset_name = "covid"
    data_path = "./datasets/covid/parsed/covid_parsed.csv"
    job_dir = context.get("job_dir")
    if job_dir is not None:
        experiment_dir = Path(job_dir)
    else:
        experiment_dir = None
    context.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "sensitive_covar": ['Branca', 'Preta', 'Amarela', 'Parda', 'Indigena'] + ['Northeast', 'North', 'Southeast', 'Central-West', 'South'], 
            "test_alternative": "greater",
            "ds_train": {
                "data_path": data_path,
                "split": "train",
                "seed": context.get("seed"),
            },
            "ds_valid": {
                "data_path": data_path,
                "split": "valid",
                "seed": context.get("seed"),
            },
            "ds_test": {
                "data_path": data_path,
                "split": "test",
                "seed": context.get("seed"),
            },
        }
    )

    
def deep_kernel_gp_config(context, kernel, num_inducing_points, dim_hidden, dim_output, depth, negative_slope,
                          dropout_rate, spectral_norm, learning_rate, batch_size, epochs, train_ratio):
    context.update(
        {"model_name": "deep_kernel_gp",
         "kernel": kernel,
         "num_inducing_points": num_inducing_points,
         "dim_hidden": dim_hidden,
         "depth": depth,
         "dim_output": dim_output,
         "negative_slope": negative_slope,
         "dropout_rate": dropout_rate,
         "spectral_norm": spectral_norm,
         "learning_rate": learning_rate,
         "batch_size": batch_size,
         "epochs": epochs,
         "train_ratio": train_ratio}
    )


def evaluation_config(context, experiment_dir, output_dir=None):
    context["experiment_dir"] = experiment_dir
    if output_dir is not None:
        context["output_dir"] = output_dir
