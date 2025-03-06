import logging

import pandas as pd
from tensordict.nn import TensorDictModuleBase

from motion_planning.envs.motion_planning import MotionPlanningEnvParams
from motion_planning.evaluate.common import evaluate_policy
from motion_planning.utils import compute_width

logger = logging.getLogger(__name__)


def scalability(
    env_params: MotionPlanningEnvParams,
    policy: TensorDictModuleBase,
    max_steps: int,
    num_episodes: int,
    num_workers: int | None = None,
):
    """
    Evaluate the scalability of the policy by varying the number of agents.

    NB:
    1. Uses density = env_params.n_agents / env_params.width^2. However, the number of agents in `env_params` is ignored.
    Instead, the number of agents is varied from [100, 200, 500, 1000] and the width is inferred from the density.
    2. env_params.scenario is ignored. Uses `clusters` scenario with `n_samples_per_cluster` in `[(1,1), (5,5), (10,10)]`.
    3. env_params.expert_policy is ignored.

    Returns:
        A pandas DataFrame containing the results. Adds extra columns:
        - `n_agents`: The number of agents.
        - `width`: The width of the environment.
        - `area`: The area of the environment.
        - `density`: The density of the environment.
        - `n_samples_per_cluster`: The number of samples per cluster (for both agents and targets).
    """
    density = env_params.n_agents / env_params.width**2
    df_list = []
    for n_agents in [100, 200, 300, 500, 700, 1000]:
        logger.info(f"Evaluating scalability for {n_agents} agents.")
        env_params.width = compute_width(n_agents, density)
        env_params.n_agents = n_agents
        df, _ = evaluate_policy(
            env_params, policy, max_steps, num_episodes, num_workers
        )
        df["n_agents"] = n_agents
        df["width"] = env_params.width
        df["area"] = env_params.width**2
        df["density"] = density
        df_list.append(df)
    return pd.concat(df_list)
