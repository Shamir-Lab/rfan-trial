# Set the environment variable
import os
import argparse
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import warnings
warnings.filterwarnings('ignore')

from src.evaluation.plotting import *
from src.benchmark import run_benchmark

HOME_PATH = "." # Put your path here

def main(dataset_name, max_acquisitions, switching_step):
    os.environ['PYTHONHASHSEED'] = str(0)
    set_seeds(0)

    benchmark_dir = Path(HOME_PATH + f"reports/{dataset_name}/{dataset_name}_benchmark") / f"ma_{max_acquisitions}"

    output_dir = benchmark_dir / f"t={switching_step}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Switch point, patient function, treatment function
    acquisition_settings = [(switching_step, "uniform", "random"),
                            (switching_step, "mu-max", "mu-max"),
                            (switching_step, "mu-pi", "policy"),
                            (switching_step, "mu-pi", "random"),
                            (switching_step, "mu-pi", "mu-max"),
                            (switching_step, "sign-tau", "policy")]

    # Run and store data
    methods = run_benchmark(acquisition_settings, max_acquisitions, dataset_name, HOME_PATH, output_dir)

    # Plots
    update_styles(switching_step)
    rename_labels = {f"t={switching_step}_uniform_random" : "RCT",
                     f"t={switching_step}_mu-max_mu-max" : r"$\alpha_{\mu-max}$",
                     f"t={switching_step}_mu-pi_policy"  : r"$\alpha_{\mu_\pi}$",
                     f"t={switching_step}_mu-pi_mu-max"  : r"$\alpha_{\mu_{\pi}-max}$",
                     f"t={switching_step}_mu-pi_random"  : r"$\alpha_{\mu_{\pi}-Unf}$"}

    plot_policy_values(Path(output_dir), methods, rename_labels=rename_labels)
    plot_policy_values(Path(output_dir), methods, fairness=True, rename_labels=rename_labels)
    plot_trial_statistics(Path(output_dir), methods, rename_labels=rename_labels)
    plot_objectives(Path(output_dir), methods, rename_labels=rename_labels)
    final_table = format_performance_table(methods, switching_step, output_dir)


    final_table['acquisition_method'] = final_table['acquisition'].str.replace(r'^t=\d+_', '')
    final_table.to_csv(benchmark_dir / "banchmark_performance.csv")
    print(f"Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RFAN benchmark experiments.")
    parser.add_argument("--dataset", type=str, default="synthetic", help="Dataset name ('synthetic', 'iwpc' or 'covid')")
    parser.add_argument("--max_acquisitions", type=int, default=30, help="Maximum number of acquisition steps (T)")
    parser.add_argument("--switching_step", type=int, default=14, help="Switching step (t*)")

    args = parser.parse_args()

    main(args.dataset, args.max_acquisitions, args.switching_step)
