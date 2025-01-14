import argparse
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List

import pandas as pd
import torch
from utils import load_model, rollout

from motion_planning.envs.motion_planning import MotionPlanning


@dataclass
class Parameters:
    n_agents: int = 100
    width: int = 10
    agent_radius: float = 0.05
    agent_margin: float = 0.05
    scenario: str = "uniform"
    checkpoint: str = "wandb://damowerko-academic/motion-planning/jwtdsmlx"
    n_trials: int = 50
    max_steps: int = 200


def evaluate(model, params):
    # evaluate the model for different agent radiuses
    env = MotionPlanning(
        n_agents=params.n_agents,
        width=params.width,
        agent_radius=params.agent_radius,
        scenario="uniform",
    )

    @torch.no_grad()
    def policy_fn(observation, centralized_state, step, graph):
        data = model.to_data(observation, centralized_state, step, graph)
        return model.model.forward_actor(data).detach().cpu().numpy()

    data, _ = rollout(
        env,
        policy_fn,
        vars(params),
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
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers to use. Will use multiprocessing if > 1.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="wandb://damowerko-academic/motion-planning/jwtdsmlx",
        help="The checkpoint to evaluate",
    )
    args = parser.parse_args()

    model, model_name = load_model(args.checkpoint)
    model = model.eval()

    executor_cls = ProcessPoolExecutor if args.n_workers > 1 else ThreadPoolExecutor
    with executor_cls(max_workers=args.n_workers) as e:
        # vary number of agents and density [agents / m^2]
        futures: List[Future] = []
        for n_agents, density in product(
            [20, 50, 100, 200, 500, 1000],
            [1.0],
        ):
            width = (n_agents / density) ** 0.5
            futures.append(
                e.submit(
                    evaluate,
                    model,
                    Parameters(
                        n_agents=n_agents, width=width, checkpoint=args.checkpoint
                    ),
                )
            )
        pd.concat([f.result() for f in futures]).to_parquet(
            Path("data") / "test_results" / model_name / "scalability.parquet"
        )

        # futures: List[Future] = []
        # for radius in [0.05, 0.1, 0.2, 0.3, 0.4]:
        #     futures.append(e.submit(evaluate, model, Parameters(agent_radius=radius)))
        # pd.concat([f.result() for f in futures]).to_parquet(
        #     Path("data") / "test_results" / model_name / "radius.parquet"
        # )


if __name__ == "__main__":
    main()
