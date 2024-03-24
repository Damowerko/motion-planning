from copy import deepcopy
from itertools import chain
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchcps.utils import add_model_specific_args

from motion_planning.models.base import MotionPlanningActorCritic
from motion_planning.rl import ReplayBuffer


class MotionPlanningTD3(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        policy_delay=2,
        action_noise=0.1,
        target_noise=0.2,
        target_clip=0.5,
        batches_per_epoch=1000,
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
        self.critics = nn.ModuleList([self.critic, deepcopy(self.critic)])
        self.critics_swa = nn.ModuleList(
            [torch.optim.swa_utils.AveragedModel(critic) for critic in self.critics]
        )
        self.automatic_optimization = False

    def configure_optimizers(self):
        return (
            torch.optim.AdamW(
                self.actor.parameters(), lr=self.lr, weight_decay=self.weight_decay
            ),
            torch.optim.AdamW(
                chain(self.critics[0].parameters(), self.critics[1].parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            ),
        )

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

    def training_step(self, data: Data, batch_idx):
        while len(self.buffer) < self.start_steps:
            self.buffer.extend(self.rollout())
        mu, _ = self.actor(data.state, data)

        opt_actor, opt_critic = self.optimizers()

        # actor optimizer
        mu = self.actor(data.state, data)[0]
        q = self.critic(data.state, mu, data)
        loss = -q.mean()  # maximize
        self.log("train/actor_loss", loss, prog_bar=True)

        opt_actor.zero_grad()
        self.manual_backward(loss)
        opt_actor.step()

        # also update swa for BOTH actor and critics since we use policy delay
        self.actor_swa.update_parameters(self.actor)
        for critic_swa, critic in zip(self.critics_swa, self.critics):
            critic_swa.update_parameters(critic)

        # critic optimizer
        # compute the target
        with torch.no_grad():
            muprime, _ = self.actor_swa(data.next_state, data)
            action = self.policy(muprime, self.target_noise, self.target_clip)
            qprime = torch.min(
                *[
                    critic_swa(data.next_state, action, data)
                    for critic_swa in self.critics_swa
                ]
            )
            target = data.reward + self.gamma * qprime
        # minimize mse to target
        q = self.critic(data.state, data.action, data)
        loss = F.mse_loss(q, target)
        self.log("train/critic_loss", loss, prog_bar=True, batch_size=data.batch_size)

        opt_critic.zero_grad()
        self.manual_backward(loss)
        opt_critic.step()

    def validation_step(self, data: Data, batch_idx):
        self.log(
            "val/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )
        self.log("val/metric", -data.reward.mean(), batch_size=data.batch_size)

    def test_step(self, data: Data, batch_idx):
        self.log(
            "test/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )

    def train_dataloader(self):
        return self._dataloader(n_episodes=1, use_buffer=True, render=False)

    def val_dataloader(self):
        return self._dataloader(n_episodes=10, use_buffer=False, render=self.render)

    def test_dataloader(self):
        return self._dataloader(n_episodes=100, use_buffer=False, render=self.render)
