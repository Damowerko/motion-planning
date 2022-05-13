from pickletools import optimize
from typing import List, Optional
from itertools import chain
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import LambdaCallback

from reconstrain.utils import auto_args
from reconstrain.models.base import MotionPlanningActorCritic
from reconstrain.rl import ReplayBuffer


@auto_args
class MotionPlanningTD3(MotionPlanningActorCritic):
    def __init__(
        self,
        policy_delay=2,
        action_noise=0.1,
        target_noise=0.2,
        target_clip=0.5,
        batches_per_epoch=100,
        buffer_size=100_000,
        start_steps=2_000,
        render=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore="kwargs")
        self.batches_per_epoch = batches_per_epoch
        self.render = render
        self.action_noise = action_noise
        self.target_noise = target_noise
        self.target_clip = target_clip
        self.start_steps = start_steps
        self.policy_delay = policy_delay

        self.buffer = ReplayBuffer[Data](buffer_size)
        self.actor_swa = torch.optim.swa_utils.AveragedModel(self.actor)
        self.critics = [self.critic, deepcopy(self.critic)]
        self.critics_swa = [
            torch.optim.swa_utils.AveragedModel(critic) for critic in self.critics
        ]

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(
                self.actor.parameters(), lr=self.lr, weight_decay=self.weight_decay
            ),
            torch.optim.AdamW(
                chain(self.critics[0].parameters(), self.critics[1].parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            ),
        ]

    def policy(
        self, mu: torch.Tensor, noise: float, noise_clip: Optional[float] = None
    ):
        eps = torch.randn_like(mu) * noise
        if noise_clip:
            eps = torch.clamp(eps, -noise_clip, noise_clip)
        return torch.clamp(mu + eps, -1, 1)

    def batch_generator(self, n_episodes=1, use_buffer=True, render=False):
        data = list(chain(*[self.rollout(render=render) for _ in range(n_episodes)]))
        self.buffer.extend(data)
        if use_buffer:
            return iter(self.buffer.sample(self.batch_size * self.batches_per_epoch))
        else:
            return iter(data)

    def training_step(self, data: Data, batch_idx, optimizer_idx):
        while len(self.buffer) < self.start_steps:
            self.buffer.extend(self.rollout())
        mu, _ = self.actor(data.x, data)
        if optimizer_idx == 0 and batch_idx % self.policy_delay == 0:
            # actor optimizer
            mu = self.actor(data.x, data)[0]
            q = self.critic(data.x, mu, data)
            loss = -q.mean()  # maximize
            self.log("train/actor_loss", loss, prog_bar=True)

            # also update swa for BOTH actor and critics since we use policy delay
            self.actor_swa.update_parameters(self.actor)
            for critic_swa, critic in zip(self.critics_swa, self.critics):
                critic_swa.update_parameters(critic)
        elif optimizer_idx == 1:
            # critic optimizer
            # compute the target
            with torch.no_grad():
                muprime, _ = self.actor_swa(data.xprime, data)
                action = self.policy(muprime, self.target_noise, self.target_clip)
                qprime = torch.min(
                    critic_swa(data.xprime, action, data)
                    for critic_swa in self.critics_swa
                )
                target = data.reward + self.gamma * qprime
            # minimize mse to target
            q = self.critic(data.x, data.action, data)
            loss = F.mse_loss(q, target)
            self.log("train/critic_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, data: Data, batch_idx):
        self.log("val/reward", data.reward.mean(), prog_bar=True)

    def test_step(self, data: Data, batch_idx):
        self.log("test/reward", data.reward.mean(), prog_bar=True)

    def train_dataloader(self):
        return self._dataloader(n_episodes=1, use_buffer=True, render=False)

    def val_dataloader(self):
        return self._dataloader(n_episodes=10, use_buffer=False, render=self.render)

    def test_dataloader(self):
        return self._dataloader(n_episodes=100, use_buffer=False, render=self.render)
