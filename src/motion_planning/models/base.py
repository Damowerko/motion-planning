import typing
from typing import Iterator, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch.distributions.normal import Normal
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_scatter import scatter_mean
from torchcps.utils import add_model_specific_args

from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.rl import ExperienceSourceDataset

from .gnn import GCN


def discounted_to_go(principal: torch.Tensor, discount: float) -> torch.Tensor:
    n = len(principal)
    rtgs = torch.zeros_like(principal)
    for i in reversed(range(n)):
        rtgs[i] = principal[i] + discount * (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


class GNNActor(nn.Module):
    def __init__(
        self,
        state_ndim: int,
        action_ndim: int,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_ndim = state_ndim
        self.action_ndim = action_ndim

        self.gnn = GCN(
            state_ndim,
            action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

        self.log_std = torch.nn.Parameter(torch.as_tensor(-0.5 * np.ones(action_ndim)))

    def forward(self, state: torch.Tensor, data: BaseData):
        action = self.gnn.forward(state, data.edge_index, data.edge_attr)
        mu = action[:, : self.action_ndim]
        sigma = torch.exp(self.log_std)
        return mu, sigma

    def distribution(self, state: torch.Tensor, data: BaseData) -> Normal:
        mu, sigma = self.forward(state, data)
        return Normal(mu, sigma)

    def log_prob(self, pi: Normal, action: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(action)

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
        if action is None:
            eps = torch.randn_like(mu)
            action = torch.tanh(mu + sigma * eps)
            assert isinstance(action, torch.Tensor)
            return torch.tanh(action)

        dist = Normal(mu, sigma)
        return dist, self.log_prob(dist, action)


class GNNCritic(nn.Module):
    def __init__(
        self,
        # state_ndim: int,
        # action_ndim: int,
        # n_agents: int,
        # n_layers: int = 2,
        # n_channels: int = 32,
        # activation: typing.Union[nn.Module, str] = "leaky_relu",
        # dropout: float = 0.0,
        state_ndim: int,
        action_ndim: int,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        # self.n_agents = n_agents
        # self.state_ndim = state_ndim
        # self.action_ndim = action_ndim
        # self.in_channels = n_agents * state_ndim + n_agents * action_ndim
        # self.mlp = gnn.MLP(
        #     in_channels=self.in_channels,
        #     hidden_channels=n_channels,
        #     out_channels=1,
        #     num_layers=n_layers,
        #     dropout=dropout,
        #     act=activation,
        #     norm=None,
        # )
        # self.state_ndim = state_ndim
        # self.action_ndim = action_ndim
        self.gnn = GCN(
            state_ndim + action_ndim,
            1,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    # def forward(
    #     self,
    #     centralized_state: torch.Tensor,
    #     action: torch.Tensor,
    #     data: BaseData,
    # ) -> torch.Tensor:
    #     centralized_state = centralized_state.reshape(data.batch_size, self.n_agents * self.state_ndim)
    #     action = action.reshape(data.batch_size, self.n_agents * self.action_ndim)
    #     x = torch.cat((centralized_state, action), dim=1)
    #     y = self.mlp.forward(x)
    #     return y.squeeze(-1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        data: BaseData,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        y = self.gnn.forward(x, data.edge_index, data.edge_attr)
        # y = scatter_mean(y, data.batch, dim=0)
        return y.squeeze(-1)


class GNNValue(nn.Module):
    def __init__(
        self,
        # state_ndim: int,
        # action_ndim: int,
        # n_agents: int,
        # n_layers: int = 2,
        # n_channels: int = 32,
        # activation: typing.Union[nn.Module, str] = "leaky_relu",
        # dropout: float = 0.0,
        state_ndim: int,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        # self.n_agents = n_agents
        # self.state_ndim = state_ndim
        # self.in_channels = n_agents * state_ndim
        # self.mlp = gnn.MLP(
        #     in_channels=self.in_channels,
        #     hidden_channels=n_channels,
        #     out_channels=1,
        #     num_layers=n_layers,
        #     dropout=dropout,
        #     act=activation,
        #     norm=None,
        # )
        # self.state_ndim = state_ndim
        self.gnn = GCN(
            state_ndim,
            1,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    # def forward(
    #     self,
    #     centralized_state: torch.Tensor,
    #     data: BaseData,
    # ) -> torch.Tensor:
    #     centralized_state = centralized_state.reshape(data.batch_size, self.n_agents * self.state_ndim)
    #     x = torch.cat((centralized_state, action), dim=1)
    #     y = self.mlp.forward(x)
    #     return y.squeeze(-1)

    def forward(
        self,
        state: torch.Tensor,
        data: BaseData,
    ) -> torch.Tensor:
        y = self.gnn.forward(state, data.edge_index, data.edge_attr)
        return y.squeeze(-1)


class GNNActorCritic(nn.Module):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        observation_ndim: int,
        # state_ndim: int,
        action_ndim: int,
        # n_agents: int,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        value: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.actor = GNNActor(
            observation_ndim,
            action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

        # self.critic = GNNCritic(
        #     state_ndim,
        #     action_ndim,
        #     n_agents,
        #     n_layers * 2,
        #     n_channels * 2,
        #     activation,
        #     dropout,
        # )
        self.critic = GNNCritic(
            observation_ndim,
            action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )
        self.critic2 = None  # Only used for TD3

        self.value = GNNValue(
            observation_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    def policy(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        return self.actor.policy(mu, sigma, action)

    def step(self, state: torch.Tensor, data: BaseData):
        with torch.no_grad():
            pi = self.actor.distribution(state, data)
            action = pi.sample()
            logp = self.actor.log_prob(pi, action)
            value = self.value(state, data)

        return action, logp, value

    def action(self, state: torch.Tensor, data: BaseData):
        with torch.no_grad():
            return self.actor(state, data)[0].cpu().numpy()


class MotionPlanningActorCritic(pl.LightningModule):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.0001,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        gamma=0.99,
        lam=0.9,
        polyak=0.995,
        max_steps=200,
        n_agents: int = 100,
        width: float = 10.0,
        agent_radius: float = 0.1,
        agent_margin: float = 0.1,
        collision_coefficient: float = 5.0,
        scenario: str = "uniform",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="kwargs")

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.polyak = polyak
        self.max_steps = max_steps
        self.dropout = dropout

        self.env = MotionPlanning(
            n_agents=n_agents,
            width=width,
            scenario=scenario,
            agent_radius=agent_radius + agent_margin,
            collision_coefficient=collision_coefficient,
        )
        self.ac = GNNActorCritic(
            self.env.observation_ndim + 1,  # Data is augmented with time
            # self.env.action_ndim * 2, # Agent and target positions
            self.env.action_ndim,
            # n_agents,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    def clip_action(self, action):
        magnitude = torch.norm(action, dim=-1)
        magnitude = torch.clip(magnitude, 0, self.env.max_vel)
        tmp = action[:, 0] + 1j * action[:, 1]  # Assumes two dimensions
        angles = torch.angle(tmp)
        action_x = (magnitude * torch.cos(angles))[:, None]
        action_y = (magnitude * torch.sin(angles))[:, None]
        return torch.cat([action_x, action_y], dim=1)

    def rollout_start(self):
        """
        Called before rollout starts.
        """
        return None

    def rollout_step(self, data: BaseData):
        """
        Called after rollout step.
        """
        data.action = self.ac.policy(data.mu, data.sigma)
        next_state, centralized_state, reward, done, _ = self.env.step(
            data.action.detach().cpu().numpy()  # type: ignore
        )
        return data, next_state, centralized_state, reward, done

    @torch.no_grad()
    def rollout(self, render=False) -> tuple[List[BaseData], List[np.ndarray]]:
        self.rollout_start()
        episode = []
        observation, centralized_state = self.env.reset()
        data = self.to_data(observation, centralized_state, 0, self.env.adjacency())
        frames = []
        for step in range(self.max_steps):
            if render:
                frames.append(self.env.render(mode="rgb_array"))

            # sample action
            data.mu, data.sigma = self.ac.actor(data.state, data)

            # take step
            data, next_state, centralized_state, reward, done = self.rollout_step(data)

            # add additional attributes
            next_data = self.to_data(
                next_state, centralized_state, step + 1, self.env.adjacency()
            )
            data.reward = torch.as_tensor(reward).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.next_state = next_data.state
            data.next_centralized_state = next_data.centralized_state
            data.done = torch.tensor(done, dtype=torch.bool, device=self.device)  # type: ignore

            episode.append(data)
            data = next_data
            if done:
                break

        return episode, frames

    def critic_loss(
        self,
        q: torch.Tensor,
        qprime: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(q, reward + torch.logical_not(done) * self.gamma * qprime)

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
                self.ac.actor.parameters(),
                lr=self.actor_lr,
                weight_decay=self.weight_decay,
            ),
            torch.optim.AdamW(
                self.ac.critic.parameters(),
                lr=self.critic_lr,
                weight_decay=self.weight_decay,
            ),
        ]

    def to_data(self, state, centralized_state, step, adjacency) -> BaseData:
        if isinstance(adjacency, list):
            data = []
            for i, adj in enumerate(adjacency):
                data.append(self.to_data(state[i], centralized_state[i], step, adj))
            return Batch.from_data_list(data)
        step = step / self.max_steps
        step = np.tile(step, state.shape[0])[:, None]
        state = np.concatenate([state, step], axis=1)
        state = torch.from_numpy(state).to(
            dtype=self.dtype, device=self.device  # type: ignore
        )
        centralized_state = torch.from_numpy(centralized_state).to(
            dtype=self.dtype, device=self.device
        )
        # assert state.shape == (self.env.n_nodes, self.env.observation_ndim)
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        edge_index = edge_index.to(dtype=torch.long, device=self.device)
        edge_weight = edge_weight.to(dtype=self.dtype, device=self.device)  # type: ignore
        return Data(
            state=state,
            centralized_state=centralized_state,
            edge_index=edge_index,
            edge_attr=edge_weight,
            num_nodes=state.shape[0],
        )

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
