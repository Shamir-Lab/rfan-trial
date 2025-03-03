# Set the environment variable
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import warnings
warnings.filterwarnings('ignore')
import shutil

from src.evaluation.plotting import *
from src.benchmark import run_benchmark

HOME_PATH = "." # Put your path here

if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = str(0)
    set_seeds(0)

    # Loop over switching time steps of designs (t* )
    max_acquisitions = 30    # number of acquisition steps, default=100
    dataset_name = "synthetic" # or synthetic

    mid_acq = int((max_acquisitions/2)-1)
    phases_prop_dfs = []

    benchmark_dir = Path(HOME_PATH + f"reports/{dataset_name}/{dataset_name}_benchmark") / f"ma_{max_acquisitions}"

    # for switching_step in [0, mid_acq, max_acquisitions-1]:
    for switching_step in [14]:
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

        # plot_convergence(Path(output_dir), methods=methods)
        plot_policy_values(Path(output_dir), methods, rename_labels=rename_labels)
        plot_policy_values(Path(output_dir), methods, fairness=True, rename_labels=rename_labels)
        plot_trial_statistics(Path(output_dir), methods, rename_labels=rename_labels)
        plot_objectives(Path(output_dir), methods, rename_labels=rename_labels)
        result_df = format_performance_table(methods, switching_step, output_dir)
        phases_prop_dfs.append(result_df)

        # Clean state files
        if os.path.exists(HOME_PATH + "hp_dkgp/"):
            shutil.rmtree(HOME_PATH + "hp_dkgp/")

    final_table = pd.concat(phases_prop_dfs, ignore_index=True)
    final_table['acquisition_method'] = final_table['acquisition'].str.replace(r'^t=\d+_', '')
    final_table.to_csv(benchmark_dir / "banchmark_performance.csv")

    final_table_agg = final_table.groupby(['acquisition_method', 'switching_step']).agg(['mean', 'std']).reset_index()
    final_table_agg.to_csv(benchmark_dir / "banchmark_performance_mean_SD.csv")
    plot_curve_over_t(benchmark_dir)