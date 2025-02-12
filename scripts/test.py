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
import torch_scatter
import tqdm
from matplotlib import pyplot as plt
from torch_geometric.data import Batch, Data
from utils import compute_width, load_model, rollout, simulation_args

from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.lightning.base import MotionPlanningActorCritic


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

    delay_parser = subparsers.add_parser("delay")
    delay_parser.add_argument("--checkpoint", type=str, required=True)

    for subparser in [test_parser, baseline_parser, delay_parser]:
        subparser.add_argument(
            "--name",
            type=str,
            default=None,
            help="Override the filenames of outputs to **/{name}.{ext}. If not provided the name is inferred from the checkpoint.",
        )
        simulation_args(subparser)
        subparser.add_argument("--n_trials", type=int, default=10)
        subparser.add_argument("--max_steps", type=int, default=200)
        subparser.add_argument("--output_images", action="store_true")

    params = vars(parser.parse_args())

    if "width" in params and params["width"] is None:
        params["width"] = compute_width(params["n_agents"], params["density"])

    if params["operation"] == "test":
        test(params)
    elif params["operation"] == "baseline":
        baseline(params)
    elif params["operation"] == "delay":
        delay(params)
    else:
        raise ValueError(f"Invalid operation {params['operation']}.")


def baseline(params: dict):
    env = MotionPlanning(
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
        initial_separation=params["initial_separation"],
        scenario=params["scenario"],
        max_vel=params["max_vel"],
        dt=params["dt"],
        collision_distance=params["collision_distance"],
        collision_coefficient=params["collision_coefficient"],
        coverage_cutoff=params["coverage_cutoff"],
        reward_sigma=params["reward_sigma"],
    )
    model, name = load_model(params["checkpoint"])
    model = model.eval()

    @torch.no_grad()
    def policy_fn(observation, positions, targets, graph, components, step):
        data = model.to_data(observation, positions, targets, graph, components)
        return model.model.forward_actor(data).detach().cpu().numpy()

    # can override the filename as an argument
    filename = name if params["name"] is None else params["name"]
    data, frames = rollout(env, policy_fn, params)
    save_results(filename, Path("data") / "test_results" / filename, data, frames)


class DelayedModel:
    def __init__(
        self,
        model: MotionPlanningActorCritic,
        comm_interval: int = 1,
        comm_frequency: int = 1,
        padding_mask: bool = True,
    ):
        """
        Args:
            model (MotionPlanningActorCritic): The model to wrap.
            comm_interval (int): Number of steps between information exchanges.
            comm_frequency (int): Number of information exchanges per step.
            padding_mask (bool): If True, output subgraphs rather than zero padding.

        """

        self.model = model
        self.initialized = False
        self.comm_interval = comm_interval
        self.comm_frequency = comm_frequency
        self.padding_mask = padding_mask
        if self.comm_interval != 1 and self.comm_frequency != 1:
            raise ValueError(
                f"Unexpected combination of {comm_interval=} and {comm_frequency=}"
            )

    def lazy_init(self, data: Data):
        self.initialized = True
        state = data.state
        self.n_agents = state.size(0)
        self.n_features = state.size(1)
        self.device = data.state.device

        # time is an NxN array that stores the most recent time that agent i has received information about agent j
        # negative values indicate that agent i has not received any information about agent j
        self.time = -torch.ones(self.n_agents, self.n_agents, device=self.device)
        # state_buffer is an NxNxF array that stores the most recent state of agent j that agent i has received
        self.state_buffer = torch.zeros(
            self.n_agents, self.n_agents, self.n_features, device=self.device
        )
        self.positions_buffer = torch.zeros(
            self.n_agents, self.n_agents, 2, device=self.device
        )

    def update_buffer(self, data: Data, step):
        self._update_self(data, step)
        if step % self.comm_interval != 0:
            return
        for _ in range(self.comm_frequency):
            self._simulate_communication(data)

    def _update_self(self, data: Data, step: int):
        N = torch.arange(self.n_agents)
        self.time.fill_diagonal_(step)
        self.state_buffer[N, N, :] = data.state
        self.positions_buffer[N, N, :] = data.positions

    def _simulate_communication(self, data: Data):
        assert data.edge_index is not None
        # Vectorized approach using torch_scatter
        src, dst = data.edge_index
        # time.shape == (n_agents, n_agents)
        # state_buffer.shape == (n_agents, n_agents, n_features)
        # 1. For each i in row, gather times from its neighbors col
        time_neighbors = self.time[dst]  # (E, n_agents)
        # 2. Find max times (and argmax inside each segment) for each i
        max_time, max_arg = torch_scatter.scatter_max(time_neighbors, src, dim=0)
        # 3. max_arg is the edge index, so convert to node index
        k = dst[max_arg]  # (n_agents, n_agents)
        # 4. Identify which j's should be updated
        mask = max_time > self.time
        # 5. Update buffers for all (i, j) where needed
        i, j = torch.nonzero(mask, as_tuple=True)
        self.time[mask] = max_time[mask]
        self.state_buffer[i, j] = self.state_buffer[k[i, j], j]
        self.positions_buffer[i, j] = self.positions_buffer[k[i, j], j]

    def delayed_data_batch(self):
        """
        Provides a batch of pytorch geomtric data objects, one per agent.
        Each agent receives information about other agents with a delay, if at all.
        Output is a Data object with the following fields:
            - state: The state of each agent (with delay).
            - positions: The positions of each agent (with delay).
            - mask_self: A mask indicating which agent in the batch is the information receiver.
            - num_nodes: The number of agents in the batch.
        """

        batch_list = []
        for i in range(self.n_agents):
            mask_self = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
            mask_self[i] = True
            data = Data(
                state=self.state_buffer[i],
                positions=self.positions_buffer[i],
                mask_self=mask_self,
                num_nodes=self.n_agents,
            )
            if self.padding_mask:
                data.padding_mask = self.time[i] > -1
            batch_list.append(data)
        return typing.cast(Data, Batch.from_data_list(batch_list))

    def __call__(self, state, positions, targets, graph, components, step: int):
        with torch.no_grad():
            if self.comm_interval == 0:
                data = self.model.to_data(state, positions, targets, graph, components)
                return self.model.model.forward_actor(data).detach().cpu().numpy()

            data = self.model.to_data(state, positions, targets, graph, components)
            if not self.initialized or step == 0:
                self.lazy_init(data)
            self.update_buffer(data, step)
            # create a batch of data objects with the state for each agent
            batch = self.delayed_data_batch()
            outputs = self.model.model.forward_actor(batch)[batch.mask_self]
            assert (
                batch.batch is not None
                and (
                    batch.batch[batch.mask_self]
                    == torch.arange(self.n_agents, device=self.device)
                ).all()
            )
            return outputs.detach().cpu().numpy()


def delay(params):
    model, name = load_model(params["checkpoint"])
    model = model.eval().cuda()
    filename = params.get("name", name) or name
    savedir = Path("data") / "test_results" / filename
    savedir.mkdir(parents=True, exist_ok=True)
    env = MotionPlanning(
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
    )
    dfs = []
    for comm_interval in tqdm.trange(11, position=1, desc="Comm interval"):
        policy_fn = DelayedModel(model, comm_interval=comm_interval, padding_mask=True)  # type: ignore
        df, _ = rollout(env, policy_fn, params, pbar=False)
        df["delay_s"] = comm_interval * env.dt
        df["n_agents"] = params["n_agents"]
        dfs.append(df)
        # write to disk after each rollout
        pd.concat(dfs).to_parquet(savedir / f"delay.parquet")


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
        iio.imwrite(path / f"{name}.mp4", np.concatenate(frames, axis=0), fps=10)
    else:
        # save the frames as individual .png files
        frames_path = path / f"{name}"
        frames_path.mkdir(parents=True)
        for i, frame in enumerate(itertools.chain(*frames)):
            iio.imwrite(frames_path / f"{i}.png", frame)


if __name__ == "__main__":
    main()
