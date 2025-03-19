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
FIGURE_HEIGHT = 2.5


def so_theme():
    return (
        plotting_context("paper", font_scale=1.0)
        | axes_style("ticks")
        | {
            "figure.figsize": (2.0, 3.5),
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
        font_scale=0.8,  # default font size is 10pt
        rc={
            "figure.figsize": (2.0, 3.5),
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
            ("mlp", "linear"): "MLP",
            ("mlp", "geometric"): "MLP",
            ("absolute", "linear"): "APE-L",
            ("absolute", "geometric"): "APE-G",
            ("rotary", "linear"): "RoPE-L",
            ("rotary", "geometric"): "RoPE-G",
        }[(x["encoding_type"], x["encoding_frequencies"])],
        axis=1,
    )
    return df


def plot_encoding_comparison(df_basic):
    max_step = 200
    return (
        so.Plot(
            data=rename_pe(df_basic)
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
        .facet("attention_window")
        .label(x="Time (s)", y="Coverage", col="$R_\\text{{att}}=$")
        .layout(size=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT), engine="tight")
        .theme(so_theme())
    )


def plot_encoding_scalability(df_scalability):
    max_step = 200
    return (
        so.Plot(
            data=rename_pe(df_scalability)
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
        .label(x="Number of Agents", y="Coverage", col="$R_\\text{{att}}=$")
        .layout(size=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT), engine="tight")
        .theme(so_theme())
    )


def plot_delay_over_time(df_delay) -> so.Plot:
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
        .add(so.Band(alpha=0.1))
        .add(so.Line(linewidth=0.5))
        .limit(y=(0.5, 1.0))
        .label(x="Time (s)", y="Coverage")
        .scale(color=so.Continuous("viridis"))
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
        .limit(y=(0.8, 0.9))
        .label(x="Delay (s)", y=f"Coverage at {max_time}s")
        .theme(so_theme())
        .on(ax)
        .plot()
    )
    legend = f.legends.pop(0)
    ax.legend(legend.legend_handles, [t.get_text() for t in legend.texts])
    plt.close(f)
    return p


def plot_comparison(df_compare, ylim=(0.5, 1.0)):
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
        .label(x="Time (s)", y="Coverage")
        .theme(so_theme())
        .on(ax)
        .plot()
    )
    legend = f.legends.pop(0)
    ax.legend(legend.legend_handles, [t.get_text() for t in legend.texts])
    plt.close(f)
    return p


def plot_initialization():
    fig, ax = plt.subplots(1, 3, figsize=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT))
    for i, (n_agents_per_cluster, n_goals_per_cluster) in enumerate(
        [(1, 1), (5, 1), (10, 5)]
    ):
        env = MotionPlanningEnv(
            scenario="clusters",
            samples_per_cluster=(n_agents_per_cluster, n_goals_per_cluster),
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

        ax[i].add_collection(LineCollection(segments, colors="gray", linewidths=0.5))
        ax[i].plot(
            unobserved_targets[:, 0], unobserved_targets[:, 1], "rx", markersize=3
        )
        ax[i].plot(observed_targets[:, 0], observed_targets[:, 1], "gx", markersize=3)
        ax[i].plot(positions[:, 0], positions[:, 1], "bo", markersize=2)
        ax[i].set_xlim(-env.width / 2, env.width / 2)
        ax[i].set_ylim(-env.width / 2, env.width / 2)
        ax[i].set_aspect("equal")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    return fig


def plot_frequencies():
    # Plot stem plots of frequencies and periods
    fig, ax = plt.subplots(
        1, 2, figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH / 2), layout="constrained"
    )
    period = 1000.0
    n_frequencies = 100

    # Linear frequencies
    frequencies_linear = linear_frequencies(period, n_frequencies) / 2 / np.pi
    periods_linear = 1 / frequencies_linear
    # Geometric frequencies
    frequencies_geom = geometric_frequencies(period, n_frequencies) / 2 / np.pi
    periods_geom = 1 / frequencies_geom

    fractional_index = np.arange(n_frequencies) / n_frequencies
    # Plot frequencies
    ax[0].plot(fractional_index, frequencies_linear, label="Linear")
    ax[0].plot(fractional_index, frequencies_geom, label="Geometric")
    ax[0].set_xlabel(r"$k/K$")
    ax[0].set_ylabel(r"$f_k \quad [\mathrm{m}^{-1}]$")
    # Plot periods
    ax[1].plot(fractional_index, periods_linear, label="Linear")
    ax[1].plot(fractional_index, periods_geom, label="Geometric")
    ax[1].set_xlabel("$k/K$")
    ax[1].set_ylabel(r"$1/f_k \quad [\mathrm{m}]$")
    ax[1].legend()
    plt.close(fig)
    return fig
