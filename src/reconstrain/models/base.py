from typing import Any, Dict, Iterator, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from reconstrain.envs.motion_planning import MotionPlanning
from reconstrain.rl import ExperienceSourceDataset
from reconstrain.utils import auto_args
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_scatter import scatter_mean


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
        architecture: str = "tag",
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

        Fs = [F_in] + [F] * n_layers + [F_out]
        layers = []
        for i in range(n_layers + 1):
            gnn_layer = {
                "tag": gnn.TAGConv(Fs[i], Fs[i + 1], K=K, normalize=True),
                "gat": gnn.GATv2Conv(Fs[i], Fs[i + 1]),
            }[architecture]
            layers += [
                (
                    gnn_layer,
                    "x, edge_index, edge_weight -> x",
                ),
                (activation_ if i < n_layers - 1 else nn.Identity(), "x -> x"),
            ]
        self.gnn = gnn.Sequential("x, edge_index, edge_weight", layers)

    def forward(self, x: torch.Tensor, data: BaseData) -> torch.Tensor:
        return self.gnn(x, data.edge_index, data.edge_weight)


class GNNActor(nn.Module):
    def __init__(
        self,
        state_ndim: int,
        action_ndim: int,
        n_nodes: int,
        F: int = 256,
        K: int = 4,
        n_layers: int = 4,
        activation: str = "leaky_relu",
        architecture: str = "tag",
        logsigma_scale: float = 3.0,
    ):
        super().__init__()
        self.state_ndim = state_ndim
        self.action_ndim = action_ndim
        self.n_nodes = n_nodes
        self.logsigma_scale = logsigma_scale
        self.gnn_mu = GNN(
            state_ndim,
            action_ndim,
            F,
            K,
            n_layers,
            activation=activation,
            architecture=architecture,
        )
        self.gnn_sigma = GNN(
            state_ndim,
            action_ndim,
            F,
            K,
            n_layers,
            activation=activation,
            architecture=architecture,
        )

    def forward(self, state: torch.Tensor, data: BaseData):
        mu = self.gnn_mu.forward(state, data)
        sigma = self.gnn_sigma.forward(state, data).abs()
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
        log_prob_corr = torch.log(1 - action ** 2 + 1e-6)
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
        n_nodes: int,
        F: int = 128,
        K: int = 2,
        n_layers: int = 2,
        activation: str = "leaky_relu",
        architecture: str = "tag",
    ):
        super().__init__()
        self.state_ndim = state_ndim
        self.action_ndim = action_ndim
        self.n_nodes = n_nodes
        self.gnn = GNN(
            state_ndim + action_ndim,
            1,
            F,
            K,
            n_layers,
            activation=activation,
            architecture=architecture,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        data: BaseData,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        y = self.gnn(x, data)
        return scatter_mean(y, data.batch, dim=0)


@auto_args
class MotionPlanningActorCritic(pl.LightningModule):
    def __init__(
        self,
        F: int = 128,
        K: int = 8,
        n_layers: int = 2,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        gamma=0.95,
        max_steps=200,
        activation: str = "leaky_relu",
        architecture: str = "tag",
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
            architecture=architecture,
        )
        self.critic = GNNCritic(
            self.env.observation_ndim,
            self.env.action_ndim,
            self.env.n_nodes,
            F=F,
            K=K,
            n_layers=n_layers,
            activation=activation,
            architecture=architecture,
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
            data.action = (
                self.actor.policy(data.mu, data.sigma) if self.training else data.mu
            )
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
        assert state.shape == (self.env.n_nodes, self.env.observation_ndim)
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
