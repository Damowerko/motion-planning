import logging

import pandas as pd
from tensordict.nn import TensorDictModuleBase

from motion_planning.envs.motion_planning import MotionPlanningEnvParams
from motion_planning.evaluate.common import evaluate_policy

logger = logging.getLogger(__name__)


def scenarios(
    env_params: MotionPlanningEnvParams,
    policy: TensorDictModuleBase,
    max_steps: int,
    num_episodes: int,
    num_workers: int | None = None,
):
    """
    Evaluate the policy on different scenarios. Uses all params in `env_params` except for `scenario`.

    Returns:
        A pandas DataFrame containing the results. Adds extra columns:
        - `n_agents`: The number of agents.
        - `scenario`: The scenario used for that trial.
    """
    df_list = []
    for scenario in ["circle", "two_lines", "gaussian_uniform", "icra"]:
        logger.info(f"Evaluating {scenario} scenario.")
        env_params.scenario = scenario
        df, _ = evaluate_policy(
            env_params, policy, max_steps, num_episodes, num_workers
        )
        df["scenario"] = scenario
        df["n_agents"] = env_params.n_agents
        df_list.append(df)
    for cluster_size in [1, 5, 10, 20, 25]:
        logger.info(
            f"Evaluating clusters scenario with {cluster_size} samples per cluster."
        )
        env_params.scenario = "clusters"
        env_params.samples_per_cluster = (cluster_size, cluster_size)
        df, _ = evaluate_policy(
            env_params, policy, max_steps, num_episodes, num_workers
        )
        df["scenario"] = f"clusters {cluster_size}"
        df["n_agents"] = env_params.n_agents
        df_list.append(df)
    return pd.concat(df_list)
