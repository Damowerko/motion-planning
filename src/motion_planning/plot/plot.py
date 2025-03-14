import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import plotting_context, axes_style
import seaborn.objects as so
from matplotlib.figure import Figure

ONE_COLUMN_WIDTH = 3.5
TWO_COLUMN_WIDTH = 7.16
LEGEND_WIDTH = 0.5
FIGURE_HEIGHT = 3.0


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
        .layout(size=(TWO_COLUMN_WIDTH, 3.0), engine="tight")
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


def plot_comparison(df_compare):
    f = Figure(figsize=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT), layout="tight")
    p = (
        so.Plot(
            df_compare,
            x="time",
            y="coverage_mean",
            ymin="coverage_min",
            ymax="coverage_max",
            color="policy",
        )
        .add(so.Band())
        .add(so.Line())
        .limit(y=(0.5, 1.0))
        .label(x="Time (s)", y="Coverage")
        .theme(so_theme())
        .on(f)
        .plot()
    )
    plt.close(f)
    return p
