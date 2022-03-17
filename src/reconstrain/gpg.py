from time import sleep
from typing import Tuple
import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from reconstrain.envs.motion_planning import GraphEnv, MotionPlanning
from reconstrain.rl import ExperienceSourceDataset
from reconstrain.utils import auto_args


@auto_args
class GPG(pl.LightningModule):
    def __init__(
        self,
        F: int = 512,
        K: int = 4,
        n_layers: int = 2,
        lr: float = 0.001,
        gamma: float = 0.95,
        entropy_weight: float = 0.1,
        n_envs: int = 32,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.gamma = gamma
        self.F = F
        self.K = K
        self.n_layers = n_layers
        self.entropy_weight = entropy_weight
        self.n_envs = n_envs

        env = gym.make("motion-planning-v0")
        assert isinstance(env, GraphEnv)
        self.env = env
        self.vec_env = make_vec_env("motion-planning-v0", n_envs=n_envs)

        # disable automatic optimization since we only take one step every epoch
        self.automatic_optimization = False

        activation = nn.ReLU()

        channels = (
            [self.env.observation_ndim]
            + [F] * (n_layers - 1)
            + [self.env.action_ndim * 2]
        )
        layers = []
        for i in range(n_layers):
            layers += [
                (
                    gnn.TAGConv(channels[i], channels[i + 1], K=K, normalize=True),
                    "x, edge_index, edge_weight -> x",
                ),
                (activation if i < n_layers - 1 else nn.Identity(), "x -> x"),
            ]
        self.model = gnn.Sequential("x, edge_index, edge_weight", layers)

    def forward(self, x, edge_index, edge_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.model(x, edge_index, edge_weight)
        x = torch.tanh(x)
        mu = x[..., : self.env.action_ndim].reshape(
            -1, self.env.n_nodes * self.env.action_ndim
        )
        logvar = x[..., self.env.action_ndim :].reshape(
            -1, self.env.n_nodes * self.env.action_ndim
        )
        sigma = torch.exp(logvar / 2)
        return mu, sigma

    def choose_action(self, mu, sigma):
        distribution = torch.distributions.MultivariateNormal(
            mu, torch.diag_embed(sigma)
        )
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy

    def to_graph_data(self, observation, adjacency) -> Data:
        if isinstance(adjacency, list):
            data = []
            for i, adj in enumerate(adjacency):
                data.append(self.to_graph_data(observation[i], adj))
            return Batch.from_data_list(data)
        x = (
            torch.from_numpy(observation)
            .to(dtype=self.dtype, device=self.device)
            .reshape(self.env.n_nodes, self.env.observation_ndim)
        )
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        edge_index = edge_index.to(dtype=torch.long, device=self.device)
        edge_weight = edge_weight.to(dtype=self.dtype, device=self.device)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    def training_step(self, batch, batch_idx):
        log_prob, entropy, R, reward, sigma = batch
        policy_loss = -(R[:, None, None] * log_prob).mean()
        entropy_loss = -self.entropy_weight * entropy.mean()
        loss = policy_loss + entropy_loss
        self.manual_backward(loss)
        self.log_dict(
            {
                "train/loss": loss,
                "train/policy_loss": policy_loss,
                "train/entropy_loss": entropy_loss,
                "train/reward": reward.mean(),
                "train/sigma": sigma.mean(),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return policy_loss

    def training_epoch_end(self, outputs):
        opt = self.optimizers()
        opt.step()
        opt.zero_grad()

    def validation_step(self, batch, batch_idx):
        log_prob, entropy, R, reward, sigma = batch
        self.log("val/reward", reward.mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        log_prob, entropy, R, reward, sigma = batch
        self.log("test/reward", reward.mean(), prog_bar=True)

    def generate_episode(self, render=False):
        env = self.env if render else self.vec_env

        # rollout on the vec env
        observation = env.reset()
        done = False
        rewards, log_probs, entropys, sigmas = [], [], [], []
        while not done:
            adjacency = env.adjacency() if render else env.env_method("adjacency")
            data = self.to_graph_data(observation, adjacency)
            mu, sigma = self(data.x, data.edge_index, data.edge_attr)
            action, log_prob, entropy = self.choose_action(mu, sigma)
            observation, reward, done, _ = env.step(action.detach().cpu().numpy())
            if not isinstance(done, bool):
                done = done.all()
            if render:
                env.render()

            rewards.append(
                torch.as_tensor(reward, dtype=self.dtype, device=self.device)
            )
            log_probs.append(log_prob)
            entropys.append(entropy)
            sigmas.append(sigma)

        # compute the discoutned cost to go
        rewards = torch.stack(rewards)
        R = torch.zeros_like(rewards)
        R[-1] = 0
        for i in range(1, len(rewards)):
            R[-i - 1] = rewards[-i] + self.gamma * R[-i]
        # standardize to improve stability
        R = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-8)
        return zip(log_probs, entropys, rewards, R, sigmas)

    def _dataloader(self, render=False):
        return DataLoader(
            ExperienceSourceDataset(self.generate_episode, render=render),
        )

    def train_dataloader(self):
        return self._dataloader()

    def val_dataloader(self):
        return self._dataloader(True)

    def test_dataloader(self):
        return self._dataloader(True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
