import argparse
import itertools
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from utils import load_model, rollout

from motion_planning.envs.motion_planning import MotionPlanning


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument(
        "operation",
        type=str,
        default="test",
        choices=[
            "test",
            "baseline",
        ],
        help="The operation to perform.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override the filenames of outputs to **/{name}.{ext}. If not provided the name is inferred from the checkpoint.",
    )

    operation = sys.argv[1]

    # common args
    group = parser.add_argument_group("Simulation")
    group.add_argument("--n_trials", type=int, default=10)

    # test specific args
    if operation in ("test",):
        group.add_argument("--checkpoint", type=str, required=True)
    # baseline specific args
    if operation == "baseline":
        group.add_argument(
            "--policy", type=str, default="c", choices=["c", "d0", "d1", "capt"]
        )
    # common args
    group.add_argument("--n_agents", type=int, default=100)
    group.add_argument(
        "--width",
        type=float,
        default=None,
        help="The width of the environment. Defaults to `(n_agents / density)**0.5`.",
    )
    group.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Number of agents per unit area if `width` is not provided.",
    )
    group.add_argument("--render", action="store_true")
    group.add_argument("--max_steps", type=int, default=200)
    group.add_argument(
        "--scenario",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian_uniform"],
    )
    group.add_argument("--agent_radius", type=float, default=0.05)
    group.add_argument("--agent_margin", type=float, default=0.05)
    group.add_argument("--collision_coefficient", type=float, default=5.0)
    group.add_argument("--output_images", action="store_true")
    group.add_argument(
        "--policy",
        type=str,
        help="The policy to test. Can be either a  c / d0 / d1 / capt will use.",
    )

    params = vars(parser.parse_args())
    if params["width"] is None:
        params["width"] = (params["n_agents"] / params["density"]) ** 0.5

    if params["operation"] == "test":
        test(params)
    elif params["operation"] == "baseline":
        baseline(params)
    else:
        raise ValueError(f"Invalid operation {params['operation']}.")


def baseline(params: dict):
    env = MotionPlanning(
        n_agents=params["n_agents"],
        width=params["width"],
        scenario=params["scenario"],
        agent_radius=params["agent_radius"] + params["agent_margin"],
        collision_coefficient=params["collision_coefficient"],
    )
    if params["policy"] == "c":
        policy_fn = lambda o, g: env.centralized_policy()
    elif params["policy"] == "d0":
        policy_fn = lambda o, g: env.decentralized_policy(0)
    elif params["policy"] == "d1":
        policy_fn = lambda o, g: env.decentralized_policy(1)
    elif params["policy"] == "capt":
        policy_fn = lambda o, g: env.capt_policy()
    else:
        raise ValueError(f"Invalid policy {params['policy']}.")

    data, frames = rollout(env, policy_fn, params, baseline=True)
    save_results(
        params["policy"],
        Path("data") / "test_results" / params["policy"],
        data,
        frames,
        output_images=False,
    )


def test(params):
    env = MotionPlanning(
        n_agents=params["n_agents"],
        width=params["width"],
        scenario=params["scenario"],
        agent_radius=params["agent_radius"] + params["agent_margin"],
        collision_coefficient=params["collision_coefficient"],
    )
    model, name = load_model(params["checkpoint"])
    model = model.eval()

    @torch.no_grad()
    def policy_fn(observation, positions, targets, graph):
        data = model.to_data(observation, positions, targets, graph)
        return model.model.forward_actor(data).detach().cpu().numpy()

    # can override the filename as an argument
    filename = params.get("name", name)
    data, frames = rollout(env, policy_fn, params)
    save_results(filename, Path("data") / "test_results" / filename, data, frames)


def test_q(params):
    env = MotionPlanning(
        n_agents=params["n_agents"],
        width=params["width"],
        scenario="q-scenario",
        agent_radius=params["agent_radius"] + params["agent_margin"],
        collision_coefficient=params["collision_coefficient"],
    )
    model, name = load_model(params["checkpoint"])
    model = model.eval()

    null_action = torch.zeros(env.n_agents, 2).to(device="cuda")

    observation, positions, targets = env.reset()
    data = model.to_data(observation, positions, targets, 0, env.adjacency())
    print(
        model.model.critic.forward(data.state, null_action, data)
        .mean()
        .detach()
        .cpu()
        .numpy()
    )


def save_results(
    name: str, path: Path, data: pd.DataFrame, frames: np.ndarray, output_images=False
):
    """
    Args:
        path (Path): The path to save the summary to.
        data (pd.DataFrame): Dataframe containing numerical info of performance at each step.
        frames (np.ndarray): An array of shape (n_trial, max_steps, H, W).
        output_images (bool): If True, save the frames as individual .png files. Otherwise, save a video.
    """
    path.mkdir(parents=True, exist_ok=True)

    data.to_parquet(path / f"{name}.parquet")

    # make a single plot of basic metrics
    metric_names = ["reward", "coverage", "collisions", "near_collisions"]
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
        "Near Collisions Mean": data.groupby("trial")["near_collisions"].sum().mean(),
    }
    with open(path / f"{name}.json", "w") as f:
        json.dump(metrics, f)

    if not output_images:
        # make a single video of all trials
        iio.imwrite(path / f"{name}.mp4", np.concatenate(frames, axis=0), fps=30)
    else:
        # save the frames as individual .png files
        frames_path = path / f"{name}"
        frames_path.mkdir(parents=True)
        for i, frame in enumerate(itertools.chain(*frames)):
            iio.imwrite(frames_path / f"{i}.png", frame)


if __name__ == "__main__":
    main()
