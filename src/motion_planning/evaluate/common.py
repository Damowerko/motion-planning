import logging
import typing
from math import ceil
from typing import Callable, Optional

import pandas as pd
import tensordict
import tensordict.nn
import torch
from numpy.typing import NDArray
from tensordict import TensorDictBase
from torchrl.envs import ParallelEnv
from torchrl.envs.transforms import Compose, StepCounter, Transform, TransformedEnv
from torchrl.record import PixelRenderTransform

from motion_planning.envs import MotionPlanningEnv, MotionPlanningEnvParams

logger = logging.getLogger(__name__)
PolicyFn = Callable[[TensorDictBase], TensorDictBase]


@torch.no_grad()
def evaluate_policy(
    env_params: MotionPlanningEnvParams,
    policy: PolicyFn,
    max_steps: int,
    num_episodes: int,
    num_workers: int | None = None,
    render: bool = False,
) -> tuple[pd.DataFrame, Optional[NDArray]]:
    """
    Evaluate a policy on an environment.

    Args:
        env_params: The environment parameters.
        policy: The policy to evaluate.
        max_steps: The maximum number of steps to run the policy for.
        num_episodes: The number of episodes to evaluate on.
        num_workers: The number of parallel workers to use. If None (default), will run all episodes in parallel.
        device: The device to use for the policy.
        render: Whether to render the environment.

    Returns:
        A tuple of a pandas DataFrame and an optional numpy array of rendered frames.
    """

    def make_env():
        transforms: list[Transform] = []
        transforms.append(StepCounter(step_count_key="step"))
        if render:
            transforms.append(PixelRenderTransform())
        return TransformedEnv(
            MotionPlanningEnv(**vars(env_params)),
            Compose(*transforms),
        )

    num_workers = num_workers or num_episodes
    if num_episodes % num_workers != 0:
        logger.warning(
            f"{num_episodes=} is not divisible by {num_workers=}. Will round up to {ceil(num_episodes / num_workers) * num_workers} episodes."
        )
    num_rollouts = ceil(num_episodes / num_workers)
    env = ParallelEnv(
        num_workers,
        make_env,
        serial_for_single=True,
        device=torch.device("cuda"),
    )
    try:
        td_list = typing.cast(
            list[TensorDictBase],
            [
                env.rollout(max_steps, policy, auto_cast_to_device=True)
                for _ in range(num_rollouts)
            ],
        )
    finally:
        if not env.is_closed:
            env.close()
        del env
    data = typing.cast(TensorDictBase, tensordict.cat(td_list, dim=0))
    df = td_to_df(data)
    if render:
        frames = td_to_frames(data)
        return df, frames
    return df, None


def evaluate_expert(
    env_params: MotionPlanningEnvParams,
    max_steps: int,
    num_episodes: int,
    num_workers: int | None = None,
    render: bool = False,
) -> tuple[pd.DataFrame, Optional[NDArray]]:
    """
    Evaluate an expert policy on an environment.

    Args:
        env_params: The environment parameters. Its `env_params.expert_policy` must not be None.
        num_episodes: The number of episodes to evaluate on.
        num_workers: The number of parallel workers to use. If None (default), will run all episodes in parallel.
        render: Whether to render the environment.

    Returns:
        A tuple of a pandas DataFrame and an optional numpy array of rendered frames.
    """
    if env_params.expert_policy is None:
        raise ValueError("Expert policy is not set in env_params.")
    policy = tensordict.nn.TensorDictModule(
        torch.nn.Identity(),
        in_keys=["expert"],
        out_keys=["action"],
    )
    return evaluate_policy(
        env_params,
        policy,
        max_steps=max_steps,
        num_episodes=num_episodes,
        num_workers=num_workers,
        render=render,
    )


def td_to_df(td: TensorDictBase) -> pd.DataFrame:
    n_trials, n_steps = td.shape
    td_selected = (
        td["next"].select("collisions", "reward", "coverage", "time", "step").cpu()
    )
    td_selected["trial"] = torch.arange(n_trials, dtype=torch.long)[:, None].expand(
        -1, n_steps
    )
    for key in td_selected.keys():
        if key in ["trial", "step", "collisions"]:
            td_selected[key] = td_selected[key].long()
        else:
            td_selected[key] = td_selected[key].double()
    return pd.DataFrame(td_selected.reshape(-1).apply(torch.squeeze).numpy())


def td_to_frames(td: TensorDictBase) -> NDArray:
    return td["pixels"].flatten(0, 1).cpu().detach().numpy()
