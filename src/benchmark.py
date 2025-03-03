import os
import shutil
from torch import cuda

from src.config import *
from src.active_learning.active_learning import *
from src.active_learning.early_stopping.active_learning_with_es import active_learner_with_es
from src.evaluation.plotting import *
from src.evaluation.policy_evaluation import *
from src.evaluation.statistical_testing import *


# Active learning
num_trials = 10                  # number of trials, default=1
step_size = 10                   # number of data points to acquire at each step, default=10
gpu_per_trial = 0
cpu_per_trial = 1
object_memory_store = 8000000000
tune_during_trial = False

# Early Stopping
early_stopping = False
interm_points = [0, .25, .5, .75 ,1]

# Synthetic data
num_examples = 10000       # number of training examples, defaul=10000
beta = 2                   # Coefficient for x effect on t, default=2.0
bimodal = False            # x sampled from bimodal distribution, default=False
sigma = 1.0                # standard deviation of random noise in y, default=1.0

# Model
model_name = "deep_kernel_gp"
kernel = "RBF"            # GP kernel
dim_hidden = 100
dim_output = 1
depth = 3                 # depth of feature extractor
num_inducing_points = 15  # Number of Deep GP Inducing Points
negative_slope = 0.0      # negative slope of leaky relu, default=-1 use elu
dropout_rate = 0.2        # dropout rate, default=0.1
spectral_norm = 0.95      # Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0
batch_size = 32           # number of examples to read during each training step, default=100 
learning_rate = 1e-3      # learning rate for gradient descent, default=1e-3
epochs = 300              # number of training epochs
train_ratio = 0.9

def evaluate(config, experiment_dir, output_dir):
    evaluation_config(config, experiment_dir=experiment_dir, output_dir=output_dir)
    pehe(experiment_dir=Path(experiment_dir), output_dir=Path(output_dir))
    calc_accuracy(Path(experiment_dir), Path(output_dir))
    policy_value(Path(experiment_dir), Path(output_dir))
    control_policy_value(Path(experiment_dir), Path(output_dir))
    fair_policy_value(config, Path(experiment_dir), Path(output_dir))
    statistical_testing(Path(experiment_dir), Path(output_dir), alpha=0.05, alternative=config.get("test_alternative"))
    if config.get("dataset_name") == "synthetic":
        plot_trial_data_and_test_tau(config, Path(experiment_dir), Path(output_dir), trial=0)

    
def run_benchmark(acquisition_settings, max_acquisitions, dataset_name, home_path, output_dir):
    job_dir = home_path + f"/reports/{dataset_name}"

    methods = []

    for setting in acquisition_settings:
        switching_step = setting[0]        # switching time steps of designs (t* )
        patient_function = setting[1]      # How to select patients for adaptive phase? ("uniform", "mu", "mu_max", "tau")
        treatment_function = setting[2]    # How to select treatments for adaptive phase? ("random", "policy", "mu_max_treat")

        acquisition_function = f"t={switching_step}_{patient_function}_{treatment_function}"  # acquistion function str
        methods.append(acquisition_function)

        # UPDATE CONFIGURATIONS
        # Initialize config dictionary manually
        config = {"n_gpu": cuda.device_count()}
        if early_stopping:
            active_learning_with_es_config(config, job_dir, num_trials, step_size, max_acquisitions, switching_step,
                                   patient_function, treatment_function, acquisition_function, tune_during_trial,
                                    interm_points, gpu_per_trial, cpu_per_trial, object_memory_store)
        else:
            active_learning_config(config, job_dir, num_trials, step_size, max_acquisitions, switching_step,
                                   patient_function, treatment_function, acquisition_function, tune_during_trial,
                                   gpu_per_trial, cpu_per_trial, object_memory_store)

        if dataset_name=="synthetic":
            synthetic_config(config, num_examples=num_examples, beta=beta, bimodal=bimodal, sigma=sigma)
        elif dataset_name=="iwpc":
            iwpc_config(config)
        elif dataset_name=="covid":
            covid_config(config)
        else:
            raise Exception("Error! Unkown dataset")
        deep_kernel_gp_config(config, kernel=kernel, num_inducing_points=num_inducing_points, dim_hidden=dim_hidden,
                              dim_output=dim_output, depth=depth, negative_slope=negative_slope,
                              dropout_rate=dropout_rate, spectral_norm=spectral_norm, learning_rate=learning_rate,
                              batch_size=batch_size, epochs=epochs, train_ratio=train_ratio)

        # Set Paths
        active_learning_path = f"ss-{step_size}_ma-{max_acquisitions}_af-{acquisition_function}_t-{tune_during_trial}/"
        data_path = f"be-{beta:.02f}_si-{sigma:.02f}/" if dataset_name=="synthetic" else ""
        deep_kernel_gp_path = f"k-{kernel}_ip-{num_inducing_points}-dh-{dim_hidden}_do-{dim_output}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
        experiment_dir = home_path + f"reports/{dataset_name}/al/{active_learning_path}{data_path}/dk_gp/{deep_kernel_gp_path}"

        for trial in range(config.get("num_trials", 1)):

            # Clean state files
            if os.path.exists(home_path + "hp_dkgp/"):
                shutil.rmtree(home_path + "hp_dkgp/")

            print(f"### Trial {trial}/{num_trials}")
            config["seed"] = trial
            set_seeds(config["seed"])
            if is_rct(config):
                run_rct(model_name=model_name,
                        config=config,
                        experiment_dir=config.get("experiment_dir"),
                        trial=trial)
            else:
                if early_stopping:
                    active_learner_with_es(model_name=model_name,
                                          config=config,
                                          experiment_dir=config.get("experiment_dir"),
                                          trial=trial)
                else:
                    active_learner(model_name=model_name,
                                  config=config,
                                  experiment_dir=config.get("experiment_dir"),
                                  trial=trial)

        print("### Evaluate!")
        config_path = output_dir / "config.json"
        with config_path.open(mode="w") as cp:
            json.dump(config, cp)

        update_styles(switching_step)
        evaluate(config, experiment_dir, output_dir)
            
    return methods