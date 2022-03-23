from time import sleep
from typing import List, Tuple
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
from collections import deque
import random


class GNNPolicy(nn.Module):
    def __init__(
        self,
        observation_ndim: int,
        action_ndim: int,
        n_nodes: int,
        F: int = 512,
        K: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.observation_ndim = observation_ndim
        self.action_ndim = action_ndim
        self.n_nodes = n_nodes

        activation = nn.ReLU()
        channels = [observation_ndim] + [F] * (n_layers - 1) + [action_ndim * 2]
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
        mu = x[..., : self.action_ndim]
        logvar = x[..., self.action_ndim :]
        mu = torch.tanh(mu)
        logvar = torch.tanh(logvar)
        sigma = torch.exp(2 * logvar)
        return mu, sigma


class MotionPlanningPolicy(pl.LightningModule):
    def __init__(
        self,
        F: int = 512,
        K: int = 4,
        n_layers: int = 2,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self.F = F
        self.K = K
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay

        self.env = MotionPlanning()
        self.policy = GNNPolicy(
            self.env.observation_ndim,
            self.env.action_ndim,
            self.env.n_nodes,
            F=F,
            K=K,
            n_layers=n_layers,
        )

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    def choose_action(self, mu, sigma, deterministic=False):
        distribution = torch.distributions.MultivariateNormal(
            mu, torch.diag_embed(sigma)
        )
        action = mu if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

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

    def save_policy(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load_policy(self, filename):
        self.policy.load_state_dict(torch.load(filename))


@auto_args
class MotionPlanningImitation(MotionPlanningPolicy):
    def __init__(
        self,
        F: int = 512,
        K: int = 4,
        n_layers: int = 2,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        buffer_size: int = 10000,
        target_policy: str = "c",
        render: bool = False,
        **kwargs,
    ):
        super().__init__(F=F, K=K, n_layers=n_layers, lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.target_policy = target_policy
        self.render = render > 0
        self.buffer = deque(maxlen=buffer_size)

    def training_step(self, data, batch_idx):
        yhat, _ = self.policy(data.x, data.edge_index, data.edge_attr)
        loss = F.mse_loss(yhat.view(data.y.shape), data.y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/reward", data.reward.mean(), prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx):
        yhat, _ = self.policy(data.x, data.edge_index, data.edge_attr)
        loss = F.mse_loss(yhat.view(data.y.shape), data.y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/reward", data.reward.mean(), prog_bar=True)
        self.log("val/metric", -data.reward.mean())
        return loss

    def test_step(self, data, batch_idx):
        yhat, _ = self.policy(data.x, data.edge_index, data.edge_attr)
        loss = F.mse_loss(yhat.view(data.y.shape), data.y)
        self.log("test/loss", loss)
        self.log("test/reward", data.reward.mean())
        return loss

    @torch.no_grad()
    def rollout(self, render=False) -> List[Data]:
        episode = []
        observation = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()

            if self.target_policy == "c":
                expert_action = self.env.centralized_policy()
            elif self.target_policy == "d0":
                expert_action = self.env.decentralized_policy(0)
            elif self.target_policy == "d1":
                expert_action = self.env.decentralized_policy(1)
            else:
                raise ValueError(f"Unknown target policy {self.target_policy}")

            data = self.to_graph_data(observation, self.env.adjacency())
            mu, _ = self.policy(data.x, data.edge_index, data.edge_attr)
            observation, reward, done, _ = self.env.step(mu.detach().cpu().numpy())

            data = self.to_graph_data(observation, self.env.adjacency())
            # since we are doing imitation want to learn action
            data.y = torch.from_numpy(expert_action).to(dtype=self.dtype, device=self.device)
            data.reward = torch.as_tensor(reward).to(dtype=self.dtype, device=self.device)
            episode.append(data)
        return episode

    def generate_batch(self, n_episodes=1, render=False, use_buffer=True):
        data = []
        for _ in range(n_episodes):
            data.extend(self.rollout(render=render))
        # no point in not using the validation data in training
        self.buffer.extend(data)

        if use_buffer:
            data = list(self.buffer)
            random.shuffle(data)
        return iter(data)

    def _dataloader(self, **kwargs):
        return DataLoader(
            ExperienceSourceDataset(self.generate_batch, **kwargs),
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return self._dataloader(use_buffer=True)

    def val_dataloader(self):
        return self._dataloader(render=self.render, use_buffer=False)

    def test_dataloader(self):
        return self._dataloader(n_episodes=100, render=self.render, use_buffer=False)


@auto_args
class MotionPlanningGPG(MotionPlanningPolicy):
    def __init__(
        self,
        F: int = 512,
        K: int = 4,
        n_layers: int = 2,
        lr: float = 0.001,
        gamma: float = 0.95,
        entropy_weight: float = 0.001,
        n_envs: int = 1,
        batch_size: int = 32,
        render: int = 0,
        **kwargs,
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
        self.batch_size = batch_size
        self.render = render > 0

        env = gym.make("motion-planning-v0")
        assert isinstance(env, GraphEnv)
        self.env = env
        self.vec_env = make_vec_env("motion-planning-v0", n_envs=n_envs)

        # disable automatic optimization since we only take one step every epoch
        self.automatic_optimization = False

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        log_prob, entropy, reward, R, sigma = batch
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
        log_prob, entropy, reward, R, sigma = batch
        self.log("val/reward", reward.mean(), prog_bar=True)
        self.log("val/metric", -reward.mean())

    def test_step(self, batch, batch_idx):
        log_prob, entropy, reward, R, sigma = batch
        self.log("test/reward", reward.mean())

    def generate_batch(self, render=False, parallel=False):
        if parallel and render:
            raise ValueError("Cannot render in parallel.")
        env = self.vec_env if parallel else self.env

        # rollout on the vec env
        observation = env.reset()
        done = False
        rewards, log_probs, entropys, sigmas = [], [], [], []
        while not done:
            adjacency = env.adjacency() if not parallel else env.env_method("adjacency")
            data = self.to_graph_data(observation, adjacency)
            mu, sigma = self(data.x, data.edge_index, data.edge_attr)
            action, log_prob, entropy = self.choose_action(mu, sigma)

            _action = action if self.training else mu
            observation, reward, done, _ = env.step(_action.detach().cpu().numpy())

            rewards.append(torch.as_tensor(reward))
            log_probs.append(log_prob)
            entropys.append(entropy)
            sigmas.append(sigma)

            if parallel:
                done = done.all()
            if render:
                env.render()

        # compute the discoutned cost to go
        rewards = torch.stack(rewards)
        R = torch.zeros_like(rewards)
        R[-1] = 0
        for i in range(1, len(rewards)):
            R[-i - 1] = rewards[-i] + self.gamma * R[-i]
        # baseline
        R = R - R.mean()
        return zip(log_probs, entropys, rewards, R, sigmas)

    def _dataloader(self, parallel=False, render=False):
        return DataLoader(
            ExperienceSourceDataset(
                self.generate_batch, render=render, parallel=parallel
            ),
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return self._dataloader(self.n_envs > 1, False)

    def val_dataloader(self):
        return self._dataloader(self.n_envs > 1, self.render)

    def test_dataloader(self):
        return self._dataloader(False, self.render)
