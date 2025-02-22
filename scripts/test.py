import argparse
import itertools
import json
import typing
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import ParallelEnv
from torchrl.envs.transforms import Compose, StepCounter, Transform, TransformedEnv
from torchrl.record import PixelRenderTransform

from motion_planning.envs.motion_planning import MotionPlanningEnv
from motion_planning.utils import compute_width, load_model, simulation_args


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()

    # common args
    subparsers = parser.add_subparsers(
        title="operation", dest="operation", required=True
    )

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--checkpoint", type=str, required=True)

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument(
        "--policy",
        type=str,
        default="c",
        choices=["d0", "d1", "d1_sq", "c", "c_sq", "capt"],
    )

    for subparser in [test_parser, baseline_parser]:
        subparser.add_argument(
            "--name",
            type=str,
            default=None,
            help="Override the filenames of outputs to **/{name}.{ext}. If not provided the name is inferred from the checkpoint.",
        )
        simulation_args(subparser)
        subparser.add_argument(
            "--n_trials", type=int, default=10, help="Number of trials to run."
        )
        subparser.add_argument("--max_steps", type=int, default=200)
        subparser.add_argument("--video", action="store_true")

    params = vars(parser.parse_args())

    if "width" in params and params["width"] is None:
        params["width"] = compute_width(params["n_agents"], params["density"])

    if params["operation"] == "test":
        test(params, baseline=False)
    elif params["operation"] == "baseline":
        test(params, baseline=True)
    else:
        raise ValueError(f"Invalid operation {params['operation']}.")


def make_env_from_params(params: dict):
    return MotionPlanningEnv(
        n_agents=params["n_agents"],
        width=params["width"],
        initial_separation=params["initial_separation"],
        scenario=params["scenario"],
        max_vel=params["max_vel"],
        dt=params["dt"],
        collision_distance=params["collision_distance"],
        collision_coefficient=params["collision_coefficient"],
        coverage_cutoff=params["coverage_cutoff"],
        reward_sigma=params["reward_sigma"],
        expert_policy=params.get("policy", None),
    )


def test(params, baseline=False):
    if baseline:
        # for baseline policies, their policy is computed during the step() of the environment
        # the resulting policy is stored in the "expert" key of the data dict
        policy = TensorDictModule(
            torch.nn.Identity(), in_keys=["expert"], out_keys=["action"]
        )
        name = params["policy"]
    else:
        model, name = load_model(params["checkpoint"])
        policy = model.model.get_policy_operator().eval()

    def make_env():
        env = make_env_from_params(params)
        transforms: list[Transform] = [StepCounter(step_count_key="step")]
        if params["video"]:
            transforms.append(PixelRenderTransform())
        return TransformedEnv(env, Compose(*transforms))

    env = ParallelEnv(
        params["n_trials"],
        make_env,
    )
    policy = TensorDictModule(
        torch.nn.Identity(), in_keys=["expert"], out_keys=["action"]
    )
    data = typing.cast(
        TensorDict, env.rollout(params["max_steps"], policy, trust_policy=True)
    )
    # can override the filename as an argument
    filename = name if params["name"] is None else params["name"]
    path = Path("data") / "test_results" / filename
    df = td_to_df(data)
    frames = td_to_frames(data) if params["video"] else None
    save_results(filename, path, df, frames)


def td_to_df(td: TensorDict) -> pd.DataFrame:
    n_trials, n_steps = td.shape
    td_selected = td["next"].select("step", "reward", "coverage", "collisions", "time")
    td_selected["trial"] = torch.arange(n_trials)[:, None].expand(-1, n_steps)
    return pd.DataFrame(td_selected.reshape(-1).apply(torch.squeeze).double().numpy())


def td_to_frames(td: TensorDict) -> NDArray:
    return td["pixels"].flatten(0, 1).cpu().detach().numpy()


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
