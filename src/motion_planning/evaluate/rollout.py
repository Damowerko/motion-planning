import typing
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix

from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.lightning.base import MotionPlanningActorCritic


class Policy(ABC):
    @abstractmethod
    def __call__(
        self,
        state: NDArray,
        positions: NDArray,
        targets: NDArray,
        graph: coo_matrix,
        components: NDArray,
        time: float,
    ):
        raise NotImplementedError


class BaselinePolicy(Policy):
    def __init__(self, env: MotionPlanning, name: str):
        self.env = env
        self.name = name

    def __call__(self, *args, **kwargs) -> NDArray:
        return self.env.baseline_policy(self.name)


class ActorCriticPolicy(Policy):
    def __init__(self, model: MotionPlanningActorCritic):
        self.model = model

    def __call__(
        self,
        state: NDArray,
        positions: NDArray,
        targets: NDArray,
        graph: coo_matrix,
        components: NDArray,
        time: float,
    ) -> NDArray:
        with torch.no_grad():
            data = self.model.to_data(
                state, positions, targets, graph, components, time
            )
            return self.model.model.forward_actor(data).detach().cpu().numpy()


def rollout_trial(
    env: MotionPlanning,
    policy: Policy,
    n_steps: int,
    render: bool,
) -> tuple[pd.DataFrame, list[NDArray]]:
    data = []
    frames = []
    observation, positions, targets = env.reset()
    for step in range(n_steps):
        action = policy(
            observation,
            positions,
            targets,
            env.adjacency(),
            env.components(),
            env.t,
        )
        observation, _, _, reward, _, _ = env.step(action)
        data.append(
            dict(
                step=step,
                reward=reward.mean(),
                coverage=env.coverage(),
                collisions=env.n_collisions(threshold=env.collision_distance),
                near_collisions=env.n_collisions(threshold=2 * env.collision_distance),
            )
        )
        if render:
            frames.append(env.render(mode="rgb_array"))  # type: ignore
    return pd.DataFrame(data), frames


def rollout(
    env: MotionPlanning,
    policy: Policy,
    n_trials: int,
    n_steps: int,
    render: bool = False,
    trial_iterator_wrapper: typing.Callable | None = None,
) -> tuple[pd.DataFrame, list[list[np.ndarray]]]:
    """
    Perform rollouts in the environment using a given policy.

    Args:
        env (MotionPlanning): The environment to perform rollouts in.
        policy (Policy): The policy function to use for selecting actions.
        n_trials (int): The number of rollouts to perform.
        n_steps (int): The number of steps to perform in each rollout.
        params (dict): Additional parameters for the rollouts.
        render (bool, optional): Whether to render the environment. Defaults to False.
        trial_iterator_wrapper (Callable, optional): A wrapper for the trial iterator. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the rewards and frames for each rollout.
        - rewards (pd.DataFrame): A Pandas dataframe with columns=['trial', 'step', 'reward', 'coverage', 'collisions', 'near_collisions'].
        - frames (np.ndarray): An array of shape (n_trial, max_steps, H, W) where H and W are the heights and widths of the rendered frames.
    """
    dfs: list[pd.DataFrame] = []
    frames: list[list[NDArray]] = []
    for trial in (
        trial_iterator_wrapper(range(n_trials))
        if trial_iterator_wrapper
        else range(n_trials)
    ):
        df_trial, frames_trial = rollout_trial(
            env,
            policy,
            n_steps,
            render,
        )
        dfs.append(df_trial.assign(trial=trial))
        frames.append(frames_trial)
    return pd.concat(dfs), frames
