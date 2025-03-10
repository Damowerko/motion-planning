import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns


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


def error_band(x, y_mean, y_se, **kwargs):
    lower = y_mean - y_se
    upper = y_mean + y_se
    plt.fill_between(x, lower, upper, alpha=0.2, **kwargs)


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


def plot_basic_comparison(df_basic):
    g = sns.FacetGrid(
        rename_pe(df_basic).sort_values(["PE", "time"]),
        col="attention_window",
        hue="PE",
        height=2.0,
        aspect=1.0,
        ylim=(0, 1),
    )
    g.map(error_band, "time", "coverage_mean", "coverage_se")
    g.map(sns.lineplot, "time", "coverage_mean")
    g.set_axis_labels("Time (s)", "Coverage")
    g.set_titles("$R_\\text{{att}}$ = {col_name}")
    g.add_legend()
    return g


def plot_scalability_comparison(df_scalability):
    g = sns.FacetGrid(
        rename_pe(df_scalability)
        .query("step == step.max()")
        .sort_values(["PE", "n_agents"]),
        col="attention_window",
        hue="PE",
        height=2.0,
        aspect=1.0,
    )
    g.map(error_band, "n_agents", "coverage_mean", "coverage_se")
    g.map(sns.scatterplot, "n_agents", "coverage_mean", s=5)
    g.map(sns.lineplot, "n_agents", "coverage_mean")

    g.add_legend()
    g.set_axis_labels("Number of Agents", "Coverage")
    g.set_titles("$R_\\text{{att}}$ = {col_name}")
    return g
