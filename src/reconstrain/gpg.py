from time import sleep
import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from reconstrain.envs.motion_planning import MotionPlanning
from reconstrain.rl import ExperienceSourceDataset
from reconstrain.utils import auto_args

@auto_args
class GPG(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        discount: float = 0.99,
        F: int = 32,
        K: int = 4,
        n_layers: int = 2,
        policy_noise: float = 1.0,
        batch_size: int = 32,
        num_workers: int = -1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.discount = discount
        self.F = F
        self.K = K
        self.n_layers = n_layers
        self.policy_noise = policy_noise
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.env = MotionPlanning()

        activation = nn.ReLU()

        channels = (
            [self.env.observation_space.shape[1]]
            + [F] * (n_layers - 1)
            + [self.env.action_space.shape[1]]
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

    def forward(self, x, edge_index, edge_weight):
        x = self.model(x, edge_index, edge_weight)
        x = torch.tanh(x) * self.env.max_accel
        return x

    def choose_action(self, x, edge_index, edge_weight, action=None):
        mu = self(x, edge_index, edge_weight)
        mu = mu.view(-1, *self.env.action_space.shape)
        if not self.training:
            return mu, None

        distribution = torch.distributions.Normal(mu, self.policy_noise)
        action = distribution.sample() if action is None else action
        logp = distribution.log_prob(action).sum(axis=[-1, -2])
        return action, logp

    def parse_observation(self, observation):
        x, graph = observation
        x = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
        edge_index, edge_weight = from_scipy_sparse_matrix(graph)
        edge_index = edge_index.to(dtype=torch.long, device=self.device)
        edge_weight = edge_weight.to(dtype=self.dtype, device=self.device)
        return x, edge_index, edge_weight

    def training_step(self, batch, batch_idx):
        _, logp = self.choose_action(batch.x, batch.edge_index, batch.edge_attr, action=batch.action)
        weight = torch.exp(logp - batch.logp).detach()
        loss = -(weight * batch.R[:, None, None] * logp).mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/reward", np.mean(batch.reward), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.log("test/reward", np.mean(batch.reward), prog_bar=True)

    def generate_episode(self, render=False):
        observation = self.env.reset()
        done = False
        actions, observations, rewards, logps = [], [], [], []
        while not done:
            observations.append(observation)
            with torch.no_grad():
                action, logp = self.choose_action(*self.parse_observation(observation))
            observation, reward, done, _ = self.env.step(action.detach().cpu().numpy())
            if render: 
                self.env.render()
            actions.append(action)
            rewards.append(reward)
            logps.append(logp)

        # assume the agents stay in the same position forever
        R = 0
        Rs = []
        for reward in rewards[::-1]:
            # compute the discoutned cost to go
            R = reward + self.hparams.discount * R
            Rs.append(R)
        Rs = torch.tensor(Rs[::-1], device=self.device, dtype=self.dtype)
        # standardize the rewards to improve stability
        Rs = (Rs - Rs.mean()) / (Rs.std() + 1e-8)

        for observation, action, logp, reward, R in zip(observations, actions, logps, rewards, Rs):
            action = action.to(dtype=self.dtype, device=self.device)
            x, edge_index, edge_weight = self.parse_observation(observation)
            yield Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weight,
                action=action,
                logp=logp,
                reward=reward,
                R=R,
            )

    def _dataloader(self, render=False):
        return DataLoader(
            ExperienceSourceDataset(self.generate_episode, render=render),
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return self._dataloader()

    def test_dataloader(self):
        return self._dataloader(True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)