from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from seaborn import plotting_context, axes_style
import seaborn.objects as so

from motion_planning.architecture.transformer import (
    geometric_frequencies,
    linear_frequencies,
)
from motion_planning.envs.motion_planning import MotionPlanningEnv

ONE_COLUMN_WIDTH = 3.5
TWO_COLUMN_WIDTH = 7.16
LEGEND_WIDTH = 0.5
FIGURE_HEIGHT = 2.25


def so_theme():
    return (
        plotting_context("paper", font_scale=1.0)
        | axes_style("ticks")
        | {
            "figure.figsize": (ONE_COLUMN_WIDTH, FIGURE_HEIGHT),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "lines.linewidth": 0.7,
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
        }
    )


def set_theme_paper():
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.0,
        rc={
            "figure.figsize": (ONE_COLUMN_WIDTH, FIGURE_HEIGHT),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "lines.linewidth": 0.7,
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
        },
    )


def error_band(x, y_mean, y_se, alpha=0.2, **kwargs):
    lower = y_mean - y_se
    upper = y_mean + y_se
    plt.fill_between(x, lower, upper, alpha=alpha, **kwargs)


def rename_pe(df):
    """
    Add a column "PE" to the dataframe describing the position encoding used.
    """

    df["PE"] = df.apply(
        lambda x: {
            ("mlp", "linear"): "MLP-PE",
            ("mlp", "geometric"): "MLP-PE",
            ("absolute", "linear"): "APE-L",
            ("absolute", "geometric"): "APE-G",
            ("rotary", "linear"): "RoPE-L",
            ("rotary", "geometric"): "RoPE-G",
        }[(x["encoding_type"], x["encoding_frequencies"])],
        axis=1,
    )
    return df


def rename_attention_window(df):
    attention_window_str = {
        0: "$\\infty$",
        250: 250,
        500: 500,
        1000: 1000,
    }
    df = df.assign(
        attention_window=df["attention_window"].map(attention_window_str)
    ).sort_values(["attention_window", "time"])
    return df


def plot_encoding_comparison(df_basic):
    max_step = 200
    return (
        so.Plot(
            data=rename_pe(rename_attention_window(df_basic))
            .query(f"step <= {max_step}")
            .sort_values(["PE", "time"]),
            x="time",
            y="coverage_mean",
            ymin="coverage_min",
            ymax="coverage_max",
            color="PE",
        )
        .add(so.Band())
        .add(so.Line())
        .limit(y=(0.5, 1.0), x=(25, max_step))
        .scale(y=so.Continuous().tick(count=6))
        .facet("attention_window")
        .label(x="$t$ (s)", y="Test Success Rate, $N=100$", col="$R_\\text{{att}}=$")
        .layout(size=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT), engine="tight")
        .theme(so_theme())
    )


def plot_encoding_scalability(df_scalability):
    max_step = 200
    return (
        so.Plot(
            data=rename_pe(rename_attention_window(df_scalability))
            .query(f"step == {max_step}")
            .sort_values(["PE", "n_agents"]),
            x="n_agents",
            y="coverage_mean",
            ymin="coverage_min",
            ymax="coverage_max",
            color="PE",
        )
        .add(so.Band())
        .add(so.Line())
        .add(so.Dot(pointsize=3))
        .facet("attention_window")
        .label(
            x="$N$ agents",
            y=f"Test Success Rate, $t={max_step}$",
            col="$R_\\text{{att}}=$",
        )
        .limit(y=(0.2, 1.0))
        .layout(size=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT), engine="tight")
        .theme(so_theme())
    )


def plot_delay_over_time(df_delay, window=None) -> so.Plot:
    # apply rolling mean to df_delay
    if window is not None:
        df_delay = (
            df_delay.sort_values("time")
            .groupby(["policy", "delay_s"])[
                ["time", "coverage_mean", "coverage_se", "coverage_min", "coverage_max"]
            ]
            .apply(lambda x: x.rolling(window=window, min_periods=1, on="time").mean())
            .reset_index()
        )
    return (
        so.Plot(
            data=df_delay,
            x="time",
            y="coverage_mean",
            ymin="coverage_min",
            ymax="coverage_max",
            color="delay_s",
        )
        .facet("policy")
        .add(so.Band(alpha=0.05))
        .add(so.Line(linewidth=0.5))
        .limit(y=(0.75, 0.9))
        .scale(
            y=so.Continuous().tick(count=4),
            color=so.Continuous("viridis"),
        )
        .label(x="$t$ (s)", y="Test Success Rate", color="$\\tau$ (s)")
        .theme(so_theme())
        .layout(size=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT), engine="tight")
    )


def plot_delay_terminal(df_delay):
    max_time = df_delay["time"].max()
    f, ax = plt.subplots(
        figsize=(ONE_COLUMN_WIDTH, FIGURE_HEIGHT), layout="constrained"
    )
    p = (
        so.Plot(
            data=df_delay.query(f"time == {max_time}"),
            x="delay_s",
            y="coverage_mean",
            ymin="coverage_min",
            ymax="coverage_max",
            color="policy",
        )
        .add(so.Line(linewidth=0.5))
        .add(so.Band(alpha=0.2))
        .add(so.Dot(pointsize=3))
        .limit(y=(0.75, 0.95))
        .scale(y=so.Continuous().tick(every=0.05))
        .label(x="Delay (s)", y=f"Test Success Rate at {max_time}s")
        .theme(so_theme())
        .on(ax)
        .plot()
    )
    legend = f.legends.pop(0)
    ax.legend(legend.legend_handles, [t.get_text() for t in legend.texts])
    plt.close(f)
    return p


def plot_comparison(df_compare, ylim=(0.5, 1.0), every=0.2):
    f, ax = plt.subplots(
        figsize=(ONE_COLUMN_WIDTH, FIGURE_HEIGHT), layout="constrained"
    )
    p = (
        so.Plot(
            df_compare,
            x="time",
            y="coverage_mean",
            ymin="coverage_min",
            ymax="coverage_max",
            color="policy",
        )
        .add(so.Line())
        .add(so.Band())
        .limit(y=ylim)
        .scale(y=so.Continuous().tick(every=every, minor=1))
        .label(x="$t$ (s)", y="Test Success Rate")
        .theme(so_theme())
        .on(ax)
        .plot()
    )
    legend = f.legends.pop(0)
    ax.legend(legend.legend_handles, [t.get_text() for t in legend.texts])
    plt.close(f)
    return p


def _plot_initialization(scenario, samples_per_cluster, ax):
    env = MotionPlanningEnv(
        scenario=scenario,
        samples_per_cluster=samples_per_cluster,
    )
    env.reset()

    positions = env.positions
    targets = env.targets

    observed_targets_mask = np.zeros(len(targets), dtype=bool)
    observed_targets_idx = env._observed_targets()[1].reshape(-1)
    observed_targets_mask[observed_targets_idx] = True

    observed_targets = targets[observed_targets_mask]
    unobserved_targets = targets[~observed_targets_mask]
    edge_index = env.edge_index

    start_pos = positions[edge_index[0]]
    end_pos = positions[edge_index[1]]
    segments = list(np.stack([start_pos, end_pos], axis=1))

    ax.add_collection(LineCollection(segments, colors="gray", linewidths=0.5))
    ax.plot(unobserved_targets[:, 0], unobserved_targets[:, 1], "rx", markersize=3)
    ax.plot(observed_targets[:, 0], observed_targets[:, 1], "gx", markersize=3)
    ax.plot(positions[:, 0], positions[:, 1], "bo", markersize=2)
    ax.set_xlim(-env.width / 2, env.width / 2)
    ax.set_ylim(-env.width / 2, env.width / 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_initialization():
    fig, ax = plt.subplots(1, 3, figsize=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT))
    for i, samples_per_cluster in enumerate([(1, 1), (5, 1), (10, 5)]):
        _plot_initialization("clusters", samples_per_cluster, ax[i])
    return fig


def plot_frequencies():
    # Plot stem plots of frequencies and periods
    fig, ax = plt.subplots(
        1, 2, figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH / 2), layout="constrained"
    )
    period = 1000.0
    n_frequencies = 100

    # Linear frequencies
    frequencies_linear = linear_frequencies(period, n_frequencies)
    periods_linear = 2 * np.pi / frequencies_linear
    # Geometric frequencies
    frequencies_geom = geometric_frequencies(period, n_frequencies)
    periods_geom = 2 * np.pi / frequencies_geom

    fractional_index = np.arange(n_frequencies) / n_frequencies
    # Plot frequencies
    ax[0].plot(fractional_index, frequencies_linear, label="Linear")
    ax[0].plot(fractional_index, frequencies_geom, label="Geometric")
    ax[0].set_xlabel(r"$k/K$")
    ax[0].set_ylabel(r"$\omega_k \quad [\mathrm{m}^{-1}]$")
    # Plot periods
    ax[1].plot(fractional_index, periods_linear, label="Linear")
    ax[1].plot(fractional_index, periods_geom, label="Geometric")
    ax[1].set_xlabel("$k/K$")
    ax[1].set_ylabel(r"$2\pi/\omega_k \quad [\mathrm{m}]$")
    ax[1].legend()
    plt.close(fig)
    return fig


SCENARIO_NAMES = {
    "circle": "Circle",
    "two_lines": "Two Lines",
    # "gaussian uniform": "Gaussian-Uniform",
    "icra": "ICRA",
    "clusters 1": "Clusters-1",
    "clusters 5": "Clusters-5",
    "clusters 10": "Clusters-10",
    "clusters 20": "Clusters-20",
    "clusters 25": "Clusters-25",
}


def plot_scenarios(df_scenario):
    df_scenario = df_scenario.assign(
        scenario=df_scenario["scenario"].map(SCENARIO_NAMES)
    )
    f, ax = plt.subplots(
        figsize=(ONE_COLUMN_WIDTH, FIGURE_HEIGHT), layout="constrained"
    )
    p = (
        so.Plot(
            df_scenario,
            x="time",
            y="coverage_mean",
            ymin="coverage_min",
            ymax="coverage_max",
            color="scenario",
        )
        # .facet("policy")
        .add(so.Line())
        .add(so.Band())
        .scale(y=so.Continuous().tick(every=0.2, minor=1))
        .limit(x=(0, 800))
        .label(x="$t$ (s)", y="Test Success Rate", color="Scenario")
        .theme(so_theme())
        .on(ax)
        .plot()
    )
    legend = f.legends.pop(0)
    ax.legend(legend.legend_handles, [t.get_text() for t in legend.texts], ncol=2)
    plt.close(f)
    return p


def plot_scenarios_initialization():
    f, ax = plt.subplots(
        2,
        len(SCENARIO_NAMES) // 2,
        figsize=(TWO_COLUMN_WIDTH, TWO_COLUMN_WIDTH / 2),
        layout="constrained",
    )
    ax = ax.flatten()
    for i, scenario in enumerate(SCENARIO_NAMES.keys()):
        scenario_name = SCENARIO_NAMES[scenario]
        if "clusters" in scenario:
            k = int(scenario.split(" ")[1])
            scenario = "clusters"
            samples_per_cluster = (k, k)
        else:
            samples_per_cluster = (None, None)
        _plot_initialization(scenario, samples_per_cluster, ax[i])
        ax[i].set_title(scenario_name)
    plt.close(f)
    return f


def plot_scenarios_terminal(df_scenarios):
    df_scenarios = df_scenarios.assign(
        scenario=df_scenarios["scenario"].map(SCENARIO_NAMES)
    )
    p = (
        so.Plot(
            df_scenarios.query("time == time.max()"),
            x="scenario",
            y="coverage_mean",
            ymin="coverage_min",
            ymax="coverage_max",
            color="policy",
        )
        .add(so.Bar(), so.Dodge())
        .layout(size=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT), engine="tight")
        .theme(so_theme())
        .plot()
    )
    return p
