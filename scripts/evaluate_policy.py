import argparse
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List

import pandas as pd
import torch
from main import rollout

from motion_planning.envs.motion_planning import MotionPlanning
from scripts.utils import load_model


@dataclass
class Parameters:
    n_agents: int = 100
    width: int = 10
    agent_radius: float = 0.05
    agent_margin: float = 0.05
    scenario: str = "uniform"
    checkpoint: str = "wandb://test-team-12/motion-planning/lo49pixb"
    n_trials: int = 50
    max_steps: int = 200


def evaluate(params):
    # evaluate the model for different agent radiuses
    env = MotionPlanning(
        n_agents=params.n_agents,
        width=params.width,
        agent_radius=params.agent_radius,
        scenario="uniform",
    )
    model, _ = load_model(params.checkpoint)
    model = model.eval()

    @torch.no_grad()
    def policy_fn(observation, centralized_state, step, graph):
        data = model.to_data(observation, centralized_state, step, graph)
        return model.model.actor.forward(data.state, data)[0].detach().cpu().numpy()

    data, _ = rollout(
        env,
        policy_fn,
        params,  # type: ignore
        pbar=False,
    )
    # add metadata
    data["n_agents"] = params.n_agents
    data["width"] = params.width
    data["area"] = params.width**2
    data["agent_radius"] = params.agent_radius
    data["agent_margin"] = params.agent_margin
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int)
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=args.n_workers) as e:
        # vary number of agents and density [agents / m^2]
        futures: List[Future] = []
        for n_agents, density in product(
            [20, 50, 100, 200, 500],
            [0.2, 0.5, 1.0, 2.0, 5.0],
        ):
            width = (n_agents / density) ** 0.5
            futures.append(
                e.submit(evaluate, Parameters(n_agents=n_agents, width=width))
            )
        pd.concat([f.result() for f in futures]).to_parquet(
            Path("data/scalability.parquet")
        )

        futures: List[Future] = []
        for radius in [0.05, 0.1, 0.2, 0.3, 0.4]:
            futures.append(e.submit(evaluate, Parameters(agent_radius=radius)))
        pd.concat([f.result() for f in futures]).to_parquet(Path("data/radius.parquet"))

        # futures: List[Future] = []
        # for n_agents in [10, 50, 100, 200, 500]:
        #     e.submit(evaluate_density, n_agents)
        # for n_agents in [10, 50, 100, 200, 500]:
        #     e.submit(evaluate_scalability, n_agents)
        #     e.submit(evaluate_scalability, n_agents)


if __name__ == "__main__":
    main()
