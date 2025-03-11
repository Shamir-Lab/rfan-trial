import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from src.evaluation.evaluation import *
from src.evaluation.policy_evaluation import *

sns.set(style="whitegrid", palette="colorblind")

params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 16,
    # "xtick.labelsize": 40,
    # "ytick.labelsize": 40,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
    "lines.markersize": 8,
    "font.family": "Cambria", 
    "font.size": 14}

plt.rcParams.update(params)

# C1 - orange, C2 - dark green, C3 - dark orange, C4- purple, C5 - light brown, C6- Pink
styles = {
    "t=16_uniform_random": ("C2", "-", "*"),
    "t=16_mu-max_mu-max": ("C9", "--", "X"),
    "t=16_mu-pi_policy": ("C3", "-", "+"),
    "t=16_tau_policy": ("C8", "-", "^"),
    "t=16_sign-tau_policy": ("C5", "-", "|"),
    "t=16_sign-tau_random": ("C0", "-", "o"),
    "t=16_mu-pi_random":  ("C4", "-", "x"),
    "t=16_mu-pi_mu-max": ("C1", "-", "s"),
}


def update_styles(switching_step):
    global styles
    new_styles = {f"t={switching_step}_uniform_random": ("C2", "-", "*"),
                  f"t={switching_step}_mu-max_mu-max": ("C9", "--", "X"),
                  f"t={switching_step}_mu-pi_policy": ("C3", "-", "+"),
                  f"t={switching_step}_tau_policy": ("C8", "-", "^"),
                  f"t={switching_step}_sign-tau_policy": ("C5", "-", "|"),
                  f"t={switching_step}_sign-tau_random": ("C0", "-", "o"), 
                  f"t={switching_step}_mu-pi_random": ("C4", "-", "x"),
                  f"t={switching_step}_mu-pi_mu-max": ("C1", "-", "s")}

    styles = new_styles


def plot_trial_data_and_test_tau(config, experiment_dir, output_dir, trial, show_plot=False):
    plt.rcParams.update({"legend.fontsize": 12,
                         "legend.title_fontsize": 14,
                         "axes.labelsize": 16,
                         "xtick.labelsize": 14,
                         "ytick.labelsize": 14,
                         "font.family": "Cambria",
                         "font.size": 14})
    acquisition_function = config.get("acquisition_function")
    D_rct, D_T, D_pool = load_data(config, experiment_dir, trial)
    x_test, tau_true, tau_pred, domain, x_limit = calc_tau_on_test(experiment_dir, trial)
    x_pool = np.array(D_pool['x'])
    x = np.array(D_T['x'])
    y = np.array(D_T['y'])
    t = np.array(D_T['t'])
    N = len(x)

    fig, ax = plt.subplots(4, 1,
                           figsize=(1080 / 200, 1920 / 200),
                           dpi=150, gridspec_kw={"height_ratios": [1, 1, 2, 2]})
    density_pool_axis = ax[0]
    density_trial_axis = ax[1]
    data_axis = ax[2]
    tau_axis = ax[3]
    control_color = "C0"
    treatment_color = styles[acquisition_function][0]

    idx_0 = np.argsort(x[t == 0].ravel())
    idx_1 = np.argsort(x[t == 1].ravel())

    # 1. Pool distribution
    _ = sns.kdeplot(x=x_pool.ravel(), color="lightsteelblue", fill=True, alpha=0.5, ax=density_pool_axis)

    _ = density_pool_axis.tick_params(axis="x", which="both", left=False, right=False, labelbottom=False)
    _ = density_pool_axis.set_xlim(x_limit)
    _ = density_pool_axis.set_ylabel("$\mathcal{D}_{pool}$")
    # density_pool_axis.set_title(f"Data Distribution - {acquisition_function} (N={N}, trial-{trial:03d})")

    # 2. Trial Distribution
    _ = sns.kdeplot(x=x[t == 0][idx_0].ravel(), color=control_color, fill=True, alpha=0.5,
                    label="Control", ax=density_trial_axis)
    _ = sns.kdeplot(x=x[t == 1][idx_1].ravel(), color=treatment_color, fill=True, alpha=0.5,
                    label="Treated", ax=density_trial_axis)

    _ = density_trial_axis.tick_params(axis="x", which="both", left=False, right=False, labelbottom=False)
    _ = density_trial_axis.set_xlim(x_limit)
    _ = density_trial_axis.set_ylim((0, 0.42))
    _ = density_trial_axis.legend(loc="upper left")
    _ = density_trial_axis.set_ylabel("$\mathcal{D}_{trial}$")

    # 3. Data
    _ = sns.scatterplot(x=x[t == 0][idx_0].ravel(),
                        y=y[t == 0][idx_0].ravel(),
                        color=control_color,
                        label="Control",
                        ax=data_axis,
                        alpha=0.5)

    _ = sns.scatterplot(x=x[t == 1][idx_1].ravel(),
                        y=y[t == 1][idx_1].ravel(),
                        color=treatment_color,
                        label="Treated",
                        ax=data_axis,
                        alpha=0.5)

    _ = data_axis.set_ylabel(r"Outcome")
    _ = data_axis.legend(loc="upper left")
    _ = data_axis.tick_params(axis="x", which="both", left=False, right=False, labelbottom=False)
    _ = data_axis.set_xlim(x_limit)
    _ = data_axis.set_ylim((-5, 10))

    # 4. TAU
    idx = np.argsort(x_test.ravel())
    # True test tau
    _ = tau_axis.plot(x_test[idx].ravel(), tau_true[idx].ravel(), color="black", lw=4,
                      ls=":", label=r"$\tau(\mathbf{x})$")
    # Predictions for domain
    tau_mean = tau_pred.mean(0)
    tau_2sigma = 2 * tau_pred.std(0)
    _ = tau_axis.plot(domain, tau_mean, color=treatment_color, lw=2, ls="-", alpha=1.0,
                      label=r"$\widehat{\tau}_{\mathbf{\omega}}(\mathbf{x})$")
    _ = tau_axis.fill_between(x=domain, y1=tau_mean - tau_2sigma, y2=tau_mean + tau_2sigma,
                              color=treatment_color, alpha=0.3)

    _ = tau_axis.set_xlabel("Covariate $\mathbf{x}$")
    _ = tau_axis.set_ylabel(r"Treatment Effect $\tau$")
    _ = tau_axis.set_xlim(x_limit)
    _ = tau_axis.set_ylim((-6, 10))
    _ = tau_axis.legend(loc="upper left")

    _ = plt.savefig(experiment_dir / f"trial_distribution_and_tau_{trial}.png", dpi=200)
    _ = plt.savefig(output_dir / f"trial_{trial}_tau_{acquisition_function}.png", dpi=200)
    if show_plot:
        plt.show()
    _ = plt.close()


def acquired_data_dist(
    x_pool,
    t_pool,
    x_acquired,
    t_acquired,
    domain,
    legend_title=None,
    file_path=None,
    plot_type='hist',
    show_plot=False
):
    plt.rcParams.update(
        {
            "text.color": "0.2",
            "font.weight": "bold",
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
            "axes.labelsize": 12,
            "axes.labelcolor": "0.2",
            "axes.labelweight": "bold",
            "xtick.labelsize": 14,
            "font.family": "Cambria",
            "font.size": 12
        }
    )
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(1920 / 300, 1080 / 300),
        dpi=150,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    density_axis = ax[0]
    acquire_axis = ax[1]
    control_color = "C0"
    treatment_color = "C1"
    function_color = "#ad8bd6"

    idx = np.argsort(x_pool.ravel())
    idx_0 = np.argsort(x_pool[t_pool == 0].ravel())
    idx_1 = np.argsort(x_pool[t_pool == 1].ravel())

    _ = density_axis.axvspan(-3.0, 2, facecolor=control_color, alpha=0.05)
    _ = density_axis.axvspan(-2, 3.0, facecolor=treatment_color, alpha=0.05)
    
    if plot_type == 'hist':
        plot_func = sns.histplot
        plot_args = {'bins': np.arange(-6, 6.04, 0.04)}  # Pass bins argument for histplot
    elif plot_type == 'kde':
        plot_func = sns.kdeplot
        plot_args = {}
    
    _ = plot_func(
        x=x_pool[t_pool == 0][idx_0].ravel(),
        color=control_color,
        fill=True,
        alpha=0.5,
        label="Control",
        ax=density_axis,
        **plot_args
    )
    _ = plot_func(
        x=x_pool[t_pool == 1][idx_1].ravel(),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label="Treated",
        ax=density_axis,
        **plot_args
    )
    _ = density_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = density_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    # _ = density_axis.set_ylabel("count")
    _ = density_axis.set_xlim([-3.0, 3.0])
    _ = density_axis.legend(
        loc="upper left", bbox_to_anchor=(1, 1.05), title="Pool Data"
    )
  
    _ = sns.despine()
    _ = acquire_axis.axvspan(-3.0, 2, facecolor=control_color, alpha=0.05)
    _ = acquire_axis.axvspan(-2, 3.0, facecolor=treatment_color, alpha=0.05)
    _ = plot_func(
        x=x_acquired[t_acquired == 0].ravel(),
        color=control_color,
        fill=True,
        alpha=0.5,
        label="Control",
        ax=acquire_axis,
        **plot_args
    )
    _ = plot_func(
        x=x_acquired[t_acquired == 1].ravel(),
        color=treatment_color,
        fill=True,
        alpha=0.5,
        label="Treated",
        ax=acquire_axis,
        **plot_args
    )
    _ = acquire_axis.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    _ = acquire_axis.tick_params(
        axis="x", which="both", left=False, right=False, labelbottom=False
    )
    # _ = acquire_axis.set_ylabel("count")
    _ = acquire_axis.set_xlim([-3.0, 3.0])
    _ = acquire_axis.legend(
        loc="upper left", bbox_to_anchor=(1, 1.05), title=legend_title
    )

    _ = plt.savefig(file_path, dpi=150)
    if show_plot:
        plt.show()
    _ = plt.close()

def plot_policy_values(output_dir, methods, fairness=False, rename_labels={}, show_plot=False):
    data = {}
    suffix = "_wc_sensitive" if fairness else ""
    for acquisition_function in methods:
        pv_path = output_dir / f"{acquisition_function}_policy_value{suffix}.json"
        with pv_path.open(mode="r") as pv_file:
            pv = json.load(pv_file)
            if fairness:
                pv = pv["values"]
        data[acquisition_function] = list(pv.values())

    df = pd.DataFrame(data)

    # Set the style of the plot
    colors = [styles[col][0] for col in df.columns]
    df = df.rename(columns=rename_labels)

    # Create the boxplot with hue
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, palette=colors)

    # Adding labels and title
    if fairness:
        plt.ylabel("WC Sensitive Policy Value")
    else:
        plt.ylabel("Policy Value")

    _ = plt.savefig(output_dir / f"policy_values{suffix}.png", dpi=150)
    if show_plot:
        plt.show()


def plot_trial_statistics(output_dir, methods, rename_labels={}, show_plot=False):
    data = {}

    for acquisition_function in methods:
        eta_path = output_dir / f"{acquisition_function}_eta.json"
        with eta_path.open(mode="r") as eta_file:
            eta_dict = json.load(eta_file)

        data[acquisition_function] = list(eta_dict.values())

    eta_df = pd.DataFrame(data)

    # Set the style of the plot
    colors = [styles[col][0] for col in eta_df.columns]
    eta_df = eta_df.rename(columns=rename_labels)

    # 1. Plot effect size
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=eta_df, palette=colors)
    plt.xlabel("Acquisition Method")
    plt.ylabel("r$\eta$")
    _ = plt.savefig(output_dir / "eta.png", dpi=150)
    if show_plot:
        plt.show()


def plot_objectives(output_dir, methods, rename_labels={}, show_plot=False):

    performance_dict = performance_table(output_dir, methods)
    obj1_df = pd.DataFrame()
    obj2_df = pd.DataFrame()
    for acquisition in performance_dict.keys():
        obj1_df[acquisition] = performance_dict[acquisition]["objective1"]
        obj2_df[acquisition] = performance_dict[acquisition]["objective2"]

    colors = [styles[col][0] for col in obj1_df.columns]
    obj1_df = obj1_df.rename(columns=rename_labels)
    obj2_df = obj2_df.rename(columns=rename_labels)

    # 1. Plot effect size
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=obj1_df, palette=colors)
    plt.xlabel("Acquisition Method")
    plt.ylabel("Objective 1")
    _ = plt.savefig(output_dir / "objective_1.png", dpi=150)
    if show_plot:
        plt.show()

    # 2. Plot eta
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=obj2_df, palette=colors)
    plt.xlabel("Acquisition Method")
    plt.ylabel("Objective 2")
    _ = plt.savefig(output_dir / "objective_2.png", dpi=150)
    if show_plot:
        plt.show()


def format_performance_table(methods, switching_step, output_dir):
    data = performance_table(Path(output_dir), methods=methods)
    dfs = []

    for acquisition, perforamce_dict in data.items():
        df = pd.DataFrame(perforamce_dict)
        df['acquisition'] = acquisition
        dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)
    result_df["switching_step"] = switching_step
    result_df.to_csv(output_dir / 'performance_table.csv')

    final_table = result_df.groupby("acquisition").agg(['mean', 'std']).round(4)
    final_table.to_csv(output_dir / 'performance_table_mean_SD.csv')

    return result_df