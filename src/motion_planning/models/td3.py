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
        noise_clip=0.5,
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
        self.noise_clip = noise_clip
        self.target_clip = target_clip
        self.start_steps = start_steps
        self.policy_delay = policy_delay

        self.buffer = ReplayBuffer[Data](buffer_size)
        self.automatic_optimization = False
        self.ac.set_num_critics(2)
        self.ac_targ = deepcopy(self.ac)
        self.critic_params = chain(self.ac.critics[0].parameters(), self.ac.critics[1].parameters())

    def critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        q1 = self.ac.critics[0].forward(state, action, data)
        q2 = self.ac.critics[1].forward(state, action, data)

        with torch.no_grad():
            pi_targ = self.ac_targ.action(next_state, data)

            eps = torch.clamp(torch.randn_like(pi_targ) * self.target_noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(pi_targ + eps, -self.target_clip, self.target_clip)

            q1_pi_targ = self.ac_targ.critics[0].forward(next_state, next_action, data)
            q2_pi_targ = self.ac_targ.critics[1].forward(next_state, next_action, data)
            q_pi_targ = min(q1_pi_targ, q2_pi_targ)
            bellman = reward + self.gamma * (1 - done) * q_pi_targ
        
        return F.mse_loss(q1, bellman) + F.mse_loss(q2, bellman)
    
    def actor_loss(
        self,
        state: torch.Tensor,
        data: Data,
    ) -> torch.Tensor:
        q1_pi = self.ac.critics[0](state, self.ac.actor(state, data), data)
        return -q1_pi.mean()

    def configure_optimizers(self):
        return (
            torch.optim.AdamW(
                self.ac.actor.parameters(), lr=self.lr, weight_decay=self.weight_decay
            ),
            torch.optim.AdamW(
                self.critic_params,
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

        opt_actor, opt_critic = self.optimizers()

        # update the critic
        opt_critic.zero_grad()
        loss_q = self.critic_loss(data.state, data.action, data.reward, data.next_state, data.done, data)
        loss_q.backward()
        opt_critic.step()

        # update the actor
        
        # freeze the Q networks
        for p in self.critic_params:
            p.requires_grad = False

        # run a gradient descent for the actor
        opt_actor.zero_grad()
        loss_pi = self.actor_loss(data.state, data)
        loss_pi.backward()
        opt_actor.step()

        # unfreeze the Q networks
        for p in self.critic_params:
            p.requires_grad = True

        # update target networks
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

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
