import pandas as pd
from motion_planning.plot.plot import (
    plot_comparison,
    plot_encoding_comparison,
    plot_encoding_scalability,
    plot_delay_over_time,
    plot_delay_terminal,
    set_theme_paper,
)
from motion_planning.plot.data import (
    df_from_tag,
    aggregate_results,
    load_baselines,
    load_comparison,
    load_delay,
    load_test,
)
from pathlib import Path


def main():
    fig_path = Path("figures/journal")

    set_theme_paper()

    tags = [
        "compare-encoding-omniscient",
        "compare-encoding-local",
        "compare-encoding-connected-mask",
    ]
    names = ["clairvoyant", "local", "masked"]

    for tag, name in zip(tags, names):
        df_basic = aggregate_results(df_from_tag(tag, test_kind="basic"))
        df_scalability = aggregate_results(df_from_tag(tag, test_kind="scalability"))
        plot_encoding_comparison(df_basic).save(
            fig_path / f"{name}-encoding-comparison.pdf"
        )
        plot_encoding_scalability(df_scalability).save(
            fig_path / f"{name}-encoding-scalability.pdf"
        )

    # Plots that show the impact of delays on coverage
    models_delay = {
        "xdbf9fux": "TF Local",
        "o5tb680f": "TF Masked",
    }
    df_delay = load_delay(models_delay)
    plot_delay_over_time(df_delay).save(fig_path / "delay-over-time.pdf")
    plot_delay_terminal(df_delay).save(fig_path / "delay-terminal.pdf")

    # Plot that compares the different policies
    baselines = ["LSAP", "DLSAP-0", "DLSAP-4", "DLSAP-8"]
    models = {
        "8hlpz45j": "TF Clairvoyant",
        "xdbf9fux": "TF Local",
        "o5tb680f": "TF Masked",
    }
    df_compare = load_comparison(baselines, models, models_delay)
    plot_comparison(df_compare).save(fig_path / "compare.pdf")


if __name__ == "__main__":
    main()
