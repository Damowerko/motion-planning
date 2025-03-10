from motion_planning.plot.plot import (
    plot_basic_comparison,
    plot_scalability_comparison,
    set_theme_paper,
)
from motion_planning.plot.data import df_from_tag, aggregate_results
from pathlib import Path


def main():
    fig_path = Path("figures/journal")

    set_theme_paper()

    tags = [
        "compare-encoding-omniscient",
        "compare-encoding-local",
        "compare-encoding-connected-mask",
    ]
    names = ["clairvoyant", "local", "connected"]

    for tag, name in zip(tags, names):
        df_basic = aggregate_results(df_from_tag(tag, test_kind="basic"))
        df_scalability = aggregate_results(df_from_tag(tag, test_kind="scalability"))

        g = plot_basic_comparison(df_basic)
        g.savefig(fig_path / f"{name}-basic.pdf")

        g = plot_scalability_comparison(df_scalability)
        g.savefig(fig_path / f"{name}-scalability.pdf")


if __name__ == "__main__":
    main()
