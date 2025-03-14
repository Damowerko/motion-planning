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
    load_comparison,
    load_delay,
)
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


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
        logger.info(f"Plotting {tag} ({name})")
        df_basic = aggregate_results(df_from_tag(tag, test_kind="basic"))
        df_scalability = aggregate_results(df_from_tag(tag, test_kind="scalability"))
        plot_encoding_comparison(df_basic).save(
            fig_path / f"{name}-encoding-comparison.pdf", bbox_inches="tight"
        )
        plot_encoding_scalability(df_scalability).save(
            fig_path / f"{name}-encoding-scalability.pdf", bbox_inches="tight"
        )

    # Plots that show the impact of delays on coverage
    logger.info("Delay plots")
    models_delay = {
        "7969mfvs": "TF Local",
        "o5tb680f": "TF Masked",
    }
    df_delay = load_delay(models_delay)
    plot_delay_over_time(df_delay).save(
        fig_path / "delay-over-time.pdf", bbox_inches="tight"
    )
    plot_delay_terminal(df_delay).save(
        fig_path / "delay-terminal.pdf", bbox_inches="tight"
    )

    # Plot that compares the different policies
    logger.info("Comparison plots")
    baselines = ["LSAP", "DLSAP-0", "DLSAP-4", "DLSAP-8"]
    models = {
        "8hlpz45j": "TF Clairvoyant",
        "7969mfvs": "TF Local",
        "o5tb680f": "TF Masked",
    }
    df_compare = load_comparison(baselines, models, models_delay)
    plot_comparison(df_compare).save(fig_path / "compare.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
