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

        # manual optimization
        self.automatic_optimization = False

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
                    gnn.TAGConv(channels[i], channels[i + 1], K=K, normalize=False),
                    "x, edge_index, edge_weight -> x",
                ),
                (activation if i < n_layers - 1 else nn.Identity(), "x -> x"),
            ]
        self.model = gnn.Sequential("x, edge_index, edge_weight", layers)

    def forward(self, x, edge_index, edge_weight):
        x = self.model(x, edge_index, edge_weight)
        x = torch.tanh(x) * self.env.max_accel
        return x

    def choose_action(self, observation):
        mu = self(*self.parse_observation(observation))
        action = torch.normal(mu, self.policy_noise)
        return action

    def parse_observation(self, observation):
        x, graph = observation
        x = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
        edge_index, edge_weight = from_scipy_sparse_matrix(graph)
        edge_index = edge_index.to(dtype=torch.long, device=self.device)
        edge_weight = edge_weight.to(dtype=self.dtype, device=self.device)
        return x, edge_index, edge_weight

    def generate_episode(self, render=False):
        observation = self.env.reset()

        actions, states, rewards = [], [], []
        done = False
        while not done:
            with torch.no_grad():
                action = self.choose_action(observation)
            observation, reward, done, _ = self.env.step(action.detach().cpu().numpy())
            if render:
                self.env.render()
            actions.append(action)
            states.append(observation)
            rewards.append(reward)

        # assume the agents stay in the same position forever
        R = rewards[-1] / (1 - self.discount)
        Rs = []
        for reward in rewards[::-1]:
            # compute the discoutned cost to go
            R = reward + self.hparams.discount * R
            Rs.append(R)
        Rs = Rs[::-1]

        for action, observation, R in zip(actions, states, Rs):
            action = action.to(dtype=self.dtype, device=self.device)
            x, edge_index, edge_weight = self.parse_observation(observation)
            R = torch.tensor(R).to(dtype=self.dtype, device=self.device)
            yield Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weight,
                action=action,
                R=R,
            )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        opt = self.optimizers()
        opt.zero_grad()

    def training_step(self, batch, batch_idx):
        ndim = np.prod(self.env.action_space.shape)
        action = batch.action.view(-1, ndim)
        
        mu = self(batch.x, batch.edge_index, batch.edge_weight).view(-1, ndim)
        cov = torch.eye(ndim, device=self.device, dtype=self.dtype) * self.policy_noise
        dist = torch.distributions.MultivariateNormal(mu, cov)

        logp = dist.log_prob(action)
        loss = -(batch.R[:, None, None] * logp).mean()
        self.manual_backward(loss)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/reward", batch.R.mean(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, *args, **kwargs):
        pass

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        opt = self.optimizers()
        opt.step()

    def train_dataloader(self):
        dataset = ExperienceSourceDataset(self.generate_episode)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        dataset = ExperienceSourceDataset(lambda: self.generate_episode(render=True))
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)