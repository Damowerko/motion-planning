import typing
from typing import Iterator, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_scatter import scatter_mean
from torchcps.gnn import GCN
from torchcps.utils import add_model_specific_args

from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.rl import ExperienceSourceDataset


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
            action_ndim * 2,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    def forward(self, state: torch.Tensor, data: BaseData):
        action = self.gnn.forward(state, data.edge_index, data.edge_attr)
        mu = action[:, : self.action_ndim]
        sigma = F.softplus(action[:, self.action_ndim :])
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
        if action is None:
            eps = torch.randn_like(mu)
            action = torch.tanh(mu + sigma * eps)
            assert isinstance(action, torch.Tensor)
            return torch.tanh(action)
        log_prob_corr = torch.log(1 - action**2 + 1e-6)
        print(log_prob_corr.shape)
        assert False  # TODO: check above
        entropy = dist.entropy()
        assert isinstance(log_prob, torch.Tensor) and isinstance(entropy, torch.Tensor)
        return action, log_prob, entropy


class GNNCritic(nn.Module):
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

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        data: BaseData,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        y = self.gnn.forward(x, data.edge_index, data.edge_attr)
        return scatter_mean(y, data.batch, dim=0)


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
        lr: float = 0.0001,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        gamma=0.95,
        max_steps=200,
        n_agents: int = 100,
        scenario: str = "uniform",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="kwargs")

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_steps = max_steps
        self.dropout = dropout

        self.env = MotionPlanning(n_agents=n_agents, scenario=scenario)
        self.actor = GNNActor(
            self.env.observation_ndim,
            self.env.action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )
        self.critic = GNNCritic(
            self.env.observation_ndim,
            self.env.action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    def rollout_start(self):
        """
        Called before rollout starts.
        """
        return None

    def rollout_step(self, data: BaseData):
        """
        Called after rollout step.
        """
        next_state, reward, done, _ = self.env.step(data.action.detach().cpu().numpy())
        return data, next_state, reward, done

    @torch.no_grad()
    def rollout(self, render=False) -> List[BaseData]:
        self.rollout_start()
        episode = []
        observation = self.env.reset()
        data = self.to_data(observation, self.env.adjacency())
        for _ in range(self.max_steps):
            if render:
                self.env.render()

            # sample action
            data.mu, data.sigma = self.actor(data.state, data)

            # take step
            data, next_state, reward, done = self.rollout_step(data)

            # add additional attributes
            next_data = self.to_data(next_state, self.env.adjacency())
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
        qprime: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(q, reward + torch.logical_not(done) * self.gamma * qprime)

    def optimizers(self):
        opts = super().optimizers()
        if not isinstance(opts, list) or len(opts) != 2:
            raise ValueError(
                "Expect exactly two optimizers: actor and critic. Double check that `configure_optimizers` returns two optimziers."
            )
        opt_actor, opt_critic = opts
        return opt_actor, opt_critic

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
        state = torch.from_numpy(state).to(
            dtype=self.dtype, device=self.device  # type: ignore
        )
        # assert state.shape == (self.env.n_nodes, self.env.observation_ndim)
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        edge_index = edge_index.to(dtype=torch.long, device=self.device)
        edge_weight = edge_weight.to(dtype=self.dtype, device=self.device)  # type: ignore
        return Data(
            state=state,
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
