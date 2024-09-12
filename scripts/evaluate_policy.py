from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from main import load_model, rollout, save_results

from motion_planning.envs.motion_planning import MotionPlanning


@dataclass
class Parameters:
    n_agents: int = 100
    width: int = 10
    agent_radius: float = 0.1
    agent_margin: float = 0.1
    scenario: str = "uniform"
    checkpoint: str = "wandb://test-team-12/motion-planning/j0pmfvt9"
    n_trials: int = 10
    max_steps: int = 200


def evaluate(name, params):
    # evaluate the model for different agent radiuses
    env = MotionPlanning(
        n_agents=params.n_agents,
        width=params.width,
        agent_radius=params.agent_radius,
        scenario="uniform",
    )
    model, model_name = load_model(params.checkpoint)
    model = model.eval()

    @torch.no_grad()
    def policy_fn(observation, centralized_state, step, graph):
        data = model.to_data(observation, centralized_state, step, graph)
        return model.ac.actor.forward(data.state, data)[0].detach().cpu().numpy()

    data, frames = rollout(
        env,
        policy_fn,
        params,  # type: ignore
        pbar=False,
    )
    save_results(
        name,
        Path("data") / "policy_evaluation" / f"{model_name}",
        data,
        frames,
    )


def evaluate_radius(agent_radius):
    params = Parameters(agent_radius=agent_radius)
    evaluate(f"radius-{agent_radius}", params)


def evaluate_density(n_agents):
    params = Parameters(n_agents=n_agents)
    evaluate(f"density-{n_agents}", params)


def evaluate_scalability(n_agents, default_width=10, default_n_agents=100):
    # density is the number of agents per unit area
    density = default_n_agents / default_width**2
    params = Parameters(n_agents=n_agents, width=(n_agents / density) ** 0.5)
    evaluate(f"scalability-{n_agents}", params)


with ProcessPoolExecutor() as e:
    # for radius in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     e.submit(evaluate_radius, radius)
    for n_agents in [10, 50, 100, 200, 500]:
        e.submit(evaluate_density, n_agents)
    for n_agents in [10, 50, 100, 200, 500]:
        e.submit(evaluate_scalability, n_agents)
