import argparse
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from utils import load_model, load_model_name, rollout

from motion_planning.envs.motion_planning import MotionPlanning


@dataclass
class Parameters:
    n_agents: int = 100
    width: int = 1000
    collision_distance: float = 2.5
    initial_separation: float = 5.0
    scenario: str = "clusters"
    checkpoint: str = "wandb://damowerko-academic/motion-planning/jwtdsmlx"
    n_trials: int = 5
    max_steps: int = 200


def evaluate(params: Parameters):
    torch.set_float32_matmul_precision("high")
    model, _ = load_model(params.checkpoint)
    model = model.eval().cuda()
    # evaluate the model for different agent radiuses
    env = MotionPlanning(
        n_agents=params.n_agents,
        width=params.width,
        collision_distance=params.collision_distance,
        initial_separation=params.initial_separation,
        scenario=params.scenario,
    )

    @torch.no_grad()
    def policy_fn(observation, positions, targets, graph, components, time):
        data = model.to_data(observation, positions, targets, graph, components, time)
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
    data["collision_distance"] = params.collision_distance
    data["initial_separation"] = params.initial_separation
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="wandb://damowerko-academic/motion-planning/jwtdsmlx",
    )
    args = parser.parse_args()

    model_name = load_model_name(args.checkpoint)

    multiprocessing.set_start_method("spawn")

    with ProcessPoolExecutor(max_workers=args.n_workers) as e:
        # vary number of agents and density [agents / m^2]
        futures: List[Future] = []
        for n_agents, density in product(
            [50, 100, 200, 500, 1000],
            [1e-4],
        ):
            width = (n_agents / density) ** 0.5
            futures.append(
                e.submit(
                    evaluate,
                    Parameters(
                        n_agents=n_agents, width=width, checkpoint=args.checkpoint
                    ),
                )
            )
        pd.concat(
            [
                f.result()
                for f in tqdm(
                    as_completed(futures), total=len(futures), desc="Evaluating"
                )
            ]
        ).to_parquet(Path("data") / "test_results" / model_name / "scalability.parquet")

        # futures: List[Future] = []
        # for radius in [0.05, 0.1, 0.2, 0.3, 0.4]:
        #     futures.append(e.submit(evaluate, model, Parameters(agent_radius=radius)))
        # pd.concat([f.result() for f in futures]).to_parquet(
        #     Path("data") / "test_results" / model_name / "radius.parquet"
        # )


if __name__ == "__main__":
    main()
