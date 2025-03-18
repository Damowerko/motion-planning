from motion_planning.plot.plot import (
    plot_comparison,
    plot_encoding_comparison,
    plot_encoding_scalability,
    plot_delay_over_time,
    plot_delay_terminal,
    plot_initialization,
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
import argparse

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-encoding", dest="plot_encoding", action="store_false")
    parser.add_argument("--no-delay", dest="plot_delay", action="store_false")
    parser.add_argument("--no-comparison", dest="plot_comparison", action="store_false")
    parser.add_argument(
        "--no-initialization", dest="plot_initialization", action="store_false"
    )
    args = parser.parse_args()

    fig_path = Path("figures/journal")

    set_theme_paper()

    if args.plot_encoding:
        tags = [
            "compare-encoding-omniscient",
            "compare-encoding-local",
            "compare-encoding-connected-mask",
        ]
        names = ["clairvoyant", "local", "masked"]

        for tag, name in zip(tags, names):
            logger.info(f"Plotting {tag} ({name})")
            df_basic = aggregate_results(df_from_tag(tag, test_kind="basic"))
            df_scalability = aggregate_results(
                df_from_tag(tag, test_kind="scalability")
            )
            plot_encoding_comparison(df_basic).save(
                fig_path / f"{name}-encoding-comparison.pdf", bbox_inches="tight"
            )
            plot_encoding_scalability(df_scalability).save(
                fig_path / f"{name}-encoding-scalability.pdf", bbox_inches="tight"
            )

    if args.plot_delay:
        # Plots that show the impact of delays on coverage
        logger.info("Delay plots")
        models_delay = {
            "7969mfvs": "TF Local",
            "khpb9hkx": "TF Masked",
        }
        df_delay = load_delay(models_delay)
        plot_delay_over_time(df_delay).save(
            fig_path / "delay-over-time.pdf", bbox_inches="tight"
        )
        plot_delay_terminal(df_delay).save(
            fig_path / "delay-terminal.pdf", bbox_inches="tight"
        )

    if args.plot_comparison:
        # Plot that compares the masked transformer with DHBA
        logger.info("Comparison plots")
        baselines = ["LSAP", "DHBA-0", "DHBA-4", "DHBA-8"]
        df_compare = load_comparison(
            baselines,
            {"khpb9hkx": "TF Masked $\\tau=0.0$"},
            {"khpb9hkx": "TF Masked, $\\tau=0.1$"},
        )
        plot_comparison(df_compare).save(
            fig_path / "compare-baseline.pdf", bbox_inches="tight"
        )

        # Plot that compares the different transformer models
        df_compare = load_comparison(
            [],
            models={
                "mixtoko2": "TF Clairvoyant",
                "7969mfvs": "TF Local, $\\tau=0.0$",
                "khpb9hkx": "TF Masked, $\\tau=0.0$",
            },
            models_delay={
                "7969mfvs": "TF Local, $\\tau=0.1$",
                "khpb9hkx": "TF Masked, $\\tau=0.1$",
            },
        )
        plot_comparison(df_compare, ylim=(0.8, 1.0)).save(
            fig_path / "compare-transformer.pdf", bbox_inches="tight"
        )

    if args.plot_initialization:
        logger.info("Rendering initial positions")
        plot_initialization().savefig(
            fig_path / "initialization.pdf", bbox_inches="tight"
        )


if __name__ == "__main__":
    main()
