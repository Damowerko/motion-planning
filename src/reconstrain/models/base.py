from typing import Dict, Iterator, List, Optional

import gym
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from stable_baselines3.common.env_util import make_vec_env
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from reconstrain.envs.motion_planning import GraphEnv, MotionPlanning
from reconstrain.rl import ExperienceSourceDataset
from reconstrain.utils import auto_args


class GNN(nn.Module):
    activations: Dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
    }

    def __init__(
        self,
        F_in: int,
        F_out: int,
        F: int,
        K: int,
        n_layers: int,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        if F_in < 1 or F_out < 1 or F < 1 or K < 1:
            raise ValueError(
                f"F_in, F_out, F, K must be positive. Got {F_in}, {F_out}, {F}, {K}."
            )
        if n_layers < 0:
            raise ValueError(f"n_layers must be non-negative. Got {n_layers}.")
        if activation not in self.activations:
            raise ValueError(f"Unknown activation {activation}.")
        activation_: nn.Module = self.activations[activation]

        Fs = [F_in] + [F] * (n_layers) + [F_out]
        layers = []
        for i in range(n_layers + 1):
            layers += [
                (
                    gnn.TAGConv(Fs[i], Fs[i + 1], K=K, normalize=True),
                    "x, edge_index, edge_weight -> x",
                ),
                (activation_ if i < n_layers - 1 else nn.Identity(), "x -> x"),
            ]
        self.gnn = gnn.Sequential("x, edge_index, edge_weight", layers)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        return self.gnn(x, edge_index, edge_weight)


class GNNActor(nn.Module):
    def __init__(
        self,
        state_ndim: int,
        action_ndim: int,
        n_nodes: int,
        F: int = 512,
        K: int = 4,
        n_layers: int = 2,
        activation: str = "leaky_relu",
        predict_sigma=True,
    ):
        super().__init__()
        self.state_ndim = state_ndim
        self.action_ndim = action_ndim
        self.n_nodes = n_nodes
        self.predict_sigma = predict_sigma
        self.gnn = GNN(
            state_ndim,
            action_ndim * (2 if predict_sigma else 1),
            F,
            K,
            n_layers,
            activation=activation,
        )

    def forward(
        self, state: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ):
        x = self.gnn(state, edge_index, edge_weight)
        mu = x[..., : self.action_ndim]
        if not self.predict_sigma:
            return mu
        logsigma = x[..., self.action_ndim :]
        sigma = torch.exp(logsigma)
        return mu, sigma

    def policy(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        """
        Sample from a Gaussian distribution.

        Args:
            mu: (batch_size, N) mean of the Gaussian distribution
            sigma: (batch_size, N) standard deviation of the Gaussian distribution.
            action: (batch_size, N) Optional action. If given returns tuple action, log_prob, entropy.

        Returns:
            action: (batch_size, N) sample action or given action.
            log_prob: (batch_size, 1) log probability of the action (if action is given).
            entropy: (batch_size, 1) entropy of the action (if action is given).
        """
        normal = torch.distributions.Normal(mu, sigma)
        dist = torch.distributions.Independent(normal, 1)
        if action is None:
            action = dist.rsample()
            assert isinstance(action, torch.Tensor)
            return action
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        assert isinstance(log_prob, torch.Tensor) and isinstance(entropy, torch.Tensor)
        return action, log_prob, entropy


class GNNCritic(nn.Module):
    def __init__(
        self,
        state_ndim: int,
        action_ndim: int,
        n_nodes: int,
        F: int = 128,
        K: int = 2,
        n_layers: int = 2,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.state_ndim = state_ndim
        self.action_ndim = action_ndim
        self.n_nodes = n_nodes
        self.gnn = GNN(
            state_ndim + action_ndim, 1, F, K, n_layers, activation=activation
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        y = self.gnn(x, edge_index, edge_weight)
        return y.view(-1, self.n_nodes).mean(-1)


@auto_args
class MotionPlanningActorCritic(pl.LightningModule):
    def __init__(
        self,
        F: int = 512,
        K: int = 4,
        n_layers: int = 2,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        gamma=0.99,
        max_steps=200,
        activation: str = "leaky_relu",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="kwargs")

        self.F = F
        self.K = K
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_steps = max_steps

        self.env = MotionPlanning()
        self.actor = GNNActor(
            self.env.observation_ndim,
            self.env.action_ndim,
            self.env.n_nodes,
            F=F,
            K=K,
            n_layers=n_layers,
            activation=activation,
        )
        self.critic = GNNCritic(
            self.env.observation_ndim,
            self.env.action_ndim,
            self.env.n_nodes,
            F=F,
            K=K,
            n_layers=n_layers,
            activation=activation,
        )

    def before_rollout_step(self, data: BaseData):
        """Modify the data object before rollout step. Override this to add custom behavior."""
        return data

    @torch.no_grad()
    def rollout(self, render=False) -> List[BaseData]:
        episode = []
        observation = self.env.reset()
        data = self.to_data(observation, self.env.adjacency())
        for _ in range(self.max_steps):
            if render:
                self.env.render()

            # sample action
            mu, sigma = self.actor(data.state, data.edge_index, data.edge_attr)
            action = self.actor.policy(mu, sigma)
            assert isinstance(action, torch.Tensor)

            data = self.before_rollout_step(data)

            # take step
            next_state, reward, done, _ = self.env.step(action.detach().cpu().numpy())
            next_data = self.to_data(next_state, self.env.adjacency())

            # add additional attributes
            data.action = action
            data.reward = torch.as_tensor(reward).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.next_state = next_data.state
            data.done = torch.tensor(done, dtype=torch.bool, device=self.device)  # type: ignore

            episode.append(data)
            data = next_data
            if done:
                break
        return episode

    def reward_to_go(self, rewards) -> torch.Tensor:
        n = len(rewards)
        rtgs = torch.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs

    def critic_loss(
        self,
        q: torch.Tensor,
        reward: torch.Tensor,
        qprime: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(q, reward + torch.logical_not(done) * self.gamma * qprime)

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(
                self.actor.parameters(), lr=self.lr, weight_decay=self.weight_decay
            ),
            torch.optim.AdamW(
                self.critic.parameters(), lr=self.lr, weight_decay=self.weight_decay
            ),
        ]

    def to_data(self, state, adjacency) -> BaseData:
        if isinstance(adjacency, list):
            data = []
            for i, adj in enumerate(adjacency):
                data.append(self.to_data(state[i], adj))
            return Batch.from_data_list(data)
        state = (
            torch.from_numpy(state)
            .to(dtype=self.dtype, device=self.device)  # type: ignore
            .reshape(-1, self.env.observation_ndim)
        )
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        edge_index = edge_index.to(dtype=torch.long, device=self.device)
        edge_weight = edge_weight.to(dtype=self.dtype, device=self.device)  # type: ignore
        return Data(state=state, edge_index=edge_index, edge_attr=edge_weight)

    def batch_generator(self, *args, **kwargs) -> Iterator:
        """
        Generate batches of data.

        Args:
            n_episodes: Number of new episodes to generate. Can be zero.
            render: Whether to render the environment as we generate samples.
            use_buffer: Whether to use a replay buffer to generate samples.
        """
        raise NotImplementedError("Should be overriden by subclasses.")

    def _dataloader(self, **kwargs):
        return DataLoader(
            ExperienceSourceDataset(self.batch_generator, **kwargs),  # type: ignore
            batch_size=self.batch_size,
        )
