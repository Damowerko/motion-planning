import argparse
import json
import logging
from pathlib import Path

import imageio.v3 as iio
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from motion_planning.envs.motion_planning import MotionPlanningEnvParams
from motion_planning.evaluate import evaluate_expert, scenarios

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=[
            "c",
            "c_sq",
            "capt",
        ]
        + [f"d{i}" for i in range(10)]
        + [f"d{i}_sq" for i in range(10)],
    )
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--no-scenarios", dest="scenarios", action="store_false")
    params = vars(parser.parse_args())

    logger.info(f"Evaluating expert policy: {params['policy']}")

    env_params = MotionPlanningEnvParams(
        expert_policy=params["policy"],
    )

    logger.info("Basic policy evaluation")
    evalutate_df, frames = evaluate_expert(
        env_params=env_params,
        max_steps=params["max_steps"],
        num_episodes=params["n_trials"],
        num_workers=params["n_workers"],
        render=params["video"],
    )
    logger.info("Saving results")
    path = Path("data") / "test_results" / params["policy"]
    path.mkdir(parents=True, exist_ok=True)
    save_results(params["policy"], path, evalutate_df, frames)

    logger.info("Scenarios evaluation")
    if params["scenarios"]:
        logger.info("OOD evaluation")
        scenarios_df = scenarios(
            env_params=env_params,
            policy=None,
            max_steps=params["max_steps"],
            num_episodes=params["n_trials"],
            num_workers=params["n_workers"],
        )
        scenarios_df.to_parquet(path / "scenarios.parquet")


def save_results(
    name: str,
    path: Path,
    data: pd.DataFrame,
    frames: NDArray | None,
):
    """
    Args:
        path (Path): The path to save the summary to.
        data (pd.DataFrame): Dataframe containing numerical info of performance at each step.
        frames (np.ndarray): An array of shape (n_trial, max_steps, H, W).
    """
    path.mkdir(parents=True, exist_ok=True)

    data.to_parquet(path / f"{name}.parquet")

    # make a single plot of basic metrics
    metric_names = ["reward", "coverage", "collisions"]
    for metric_name in metric_names:
        sns.relplot(data=data, x="step", y=metric_name, hue="trial", kind="line")
        plt.xlabel("Step")
        plt.ylabel(f"{metric_name.replace('_', ' ').capitalize()}")
        plt.savefig(path / f"{metric_name}_{name}.png")

    # summary metrics
    metrics = {
        "Reward Mean": data["reward"].mean(),
        "Reward Std": data["reward"].std(),
        "Coverage Mean": data["coverage"].mean(),
        "Coverage Std": data["coverage"].std(),
        # Sum over step but mean over trials
        "Collisions Mean": data.groupby("trial")["collisions"].sum().mean(),
    }
    with open(path / f"{name}.json", "w") as f:
        json.dump(metrics, f)

    if frames is not None:
        # make a single video of all trials
        iio.imwrite(path / f"{name}.mp4", frames, fps=40)


if __name__ == "__main__":
    main()
