from typing import List

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torchcps.utils import add_model_specific_args

from motion_planning.architecture.base import ActorCritic
from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.rl import ExperienceSourceDataset, ReplayBuffer


class MotionPlanningActorCritic(pl.LightningModule):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: ActorCritic,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.0001,
        weight_decay: float = 0.0,
        batch_size: int = 100,
        gamma=0.99,
        max_steps=200,
        buffer_size: int = 10_000,
        # MotionPlanning environment parameters
        n_agents: int = 100,
        width: float = 1000.0,
        scenario: str = "uniform",
        collision_distance: float = 2.5,
        initial_separation: float = 5.0,
        collision_coefficient: float = 5.0,
        # Noise parameters
        noise: float = 0.05,
        noise_clip: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "kwargs"])

        self.model = model
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_steps = max_steps
        self.collision_distance = collision_distance
        self.initial_separation = initial_separation
        self.collision_coefficient = collision_coefficient
        self.noise = noise
        self.noise_clip = noise_clip
        self.buffer = ReplayBuffer[tuple[Data, Data]](buffer_size)
        self.env = MotionPlanning(
            n_agents=n_agents,
            width=width,
            scenario=scenario,
            collision_distance=collision_distance,
            initial_separation=initial_separation,
            collision_coefficient=collision_coefficient,
        )

    def policy(self, mu: torch.Tensor):
        """
        Sample policy from a normal distribution centered at mu and with standard devation `self.noise`.
        Noise will be clipped to `self.noise_clip` and the action will be clipped to the unit ball.
        """
        sigma = torch.ones_like(mu) * self.noise
        noise = torch.normal(0, sigma).clip(-self.noise_clip, self.noise_clip)
        action = mu + noise
        return self.clip_action(action)

    def critic_loss(
        self,
        q: torch.Tensor,
        next_q: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = done.size(0)
        N = q.size(0) // batch_size
        done = done[:, None].expand(batch_size, N).reshape(-1)
        return F.mse_loss(q, reward + torch.logical_not(done) * self.gamma * next_q)

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Clip action to the unit ball.
        """
        magnitude = torch.norm(action, dim=-1)
        mask = magnitude > 1.0
        # need to use torch.where to avoid in-place operations
        action = torch.where(mask[..., None], action / magnitude[..., None], action)
        return action

    def optimizers(self):
        opts = super().optimizers()
        if not isinstance(opts, list):
            raise ValueError(
                "Expected a list of optimizers: an actor and multiple critics. Double check that `configure_optimizers` returns multiple optimziers."
            )
        return opts

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(
                self.model.actor.parameters(),
                lr=self.actor_lr,
                weight_decay=self.weight_decay,
            ),
            torch.optim.AdamW(
                self.model.critic.parameters(),
                lr=self.critic_lr,
                weight_decay=self.weight_decay,
            ),
        ]

    def to_data(self, state, positions, targets, adjacency, components, time) -> Data:
        if isinstance(adjacency, list):
            data = []
            for i, adj in enumerate(adjacency):
                data.append(
                    self.to_data(
                        state[i], positions[i], targets[i], adj, components, time
                    )
                )
            return Batch.from_data_list(data)  # type: ignore
        state = torch.from_numpy(state).to(
            dtype=self.dtype, device=self.device  # type: ignore
        )
        positions = torch.from_numpy(positions).to(
            dtype=self.dtype, device=self.device  # type: ignore
        )
        targets = torch.from_numpy(targets).to(
            dtype=self.dtype, device=self.device  # type: ignore
        )
        components = torch.from_numpy(components).to(
            dtype=torch.long, device=self.device  # type: ignore
        )
        # assert state.shape == (self.env.n_nodes, self.env.observation_ndim)
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        edge_index = edge_index.to(dtype=torch.long, device=self.device)
        edge_weight = edge_weight.to(dtype=self.dtype, device=self.device)  # type: ignore
        return Data(
            state=state,
            positions=positions,
            targets=targets,
            components=components,
            time=time,
            edge_index=edge_index,
            edge_attr=edge_weight,
            num_nodes=state.shape[0],
        )

    def rollout_start(self):
        """
        Called before rollout starts.
        """
        return None

    def rollout_action(self, data: Data):
        """
        Choose an action to take in a rollout step.

        Returns:
            Modified data with data.action set to the action. Can set other fields as well.
        """
        data.mu = self.model.forward_actor(data)
        if self.training:
            data.action = self.policy(data.mu)
        else:
            data.action = self.clip_action(data.mu)
        return data

    def rollout_step(self, data: Data):
        """
        Take a step in the environment.

        Returns:
            next_data: The next data.
            reward: The reward.
            done: Whether the episode is done.
            coverage: The coverage.
            n_collisions: The number of collisions.
        """
        data = self.rollout_action(data)
        next_state, next_positions, next_targets, reward, done, _ = self.env.step(
            data.action.detach().cpu().numpy()  # type: ignore
        )
        next_data = self.to_data(
            next_state,
            next_positions,
            next_targets,
            self.env.adjacency(),
            self.env.components(),
            self.env.t,
        )
        coverage = self.env.coverage()
        n_collisions = self.env.n_collisions(threshold=self.collision_distance)
        return (
            data,
            next_data,
            reward,
            done,
            coverage,
            n_collisions,
        )

    @torch.no_grad()
    def rollout(self, render=False) -> tuple[List[Data], List[np.ndarray]]:
        self.rollout_start()
        episode = []
        observation, positions, targets = self.env.reset()
        data = self.to_data(
            observation,
            positions,
            targets,
            self.env.adjacency(),
            self.env.components(),
            self.env.t,
        )
        frames = []
        for step in range(self.max_steps):
            if render:
                frames.append(self.env.render(mode="rgb_array"))

            # take step
            (
                data,
                next_data,
                reward,
                done,
                coverage,
                n_collisions,
            ) = self.rollout_step(data)

            # add additional attributes
            data = data.clone()
            data.step = step
            data.reward = torch.as_tensor(reward).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.coverage = torch.tensor([coverage]).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.n_collisions = torch.tensor([n_collisions]).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.done = torch.tensor(done, dtype=torch.bool, device=self.device)  # type: ignore

            episode.append((data, next_data))
            data = next_data.clone()
            if done:
                break

        return episode, frames

    def batch_generator(
        self, n_episodes=1, render=False, use_buffer=True, training=True
    ):
        # set model to appropriate mode
        _training = self.training
        self.train(training)

        data = []
        for _ in range(n_episodes):
            episode, frames = self.rollout(render=render)
            data.extend(episode)
        if use_buffer:
            self.buffer.extend(data)
            data = self.buffer.collect(shuffle=True)

        # reset model to original mode
        self.train(_training)

        return iter(data)

    def _dataloader(self, **kwargs):
        return DataLoader(
            ExperienceSourceDataset(self.batch_generator, **kwargs),  # type: ignore
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return self._dataloader(
            n_episodes=10, render=False, use_buffer=True, training=True
        )

    def val_dataloader(self):
        return self._dataloader(
            n_episodes=1, render=False, use_buffer=False, training=False
        )

    def test_dataloader(self):
        return self._dataloader(
            n_episodes=100, render=False, use_buffer=False, training=False
        )
