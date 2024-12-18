import random
from copy import deepcopy

import numpy as np

from motion_planning.models.base import *
from motion_planning.rl import ReplayBuffer


class MotionPlanningDDPG(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        buffer_size: int = 100_000,
        start_steps: int = 2_000,
        render: bool = False,
        **kwargs,
    ):
        """
        Args:
            buffer_size: size of the replay buffer
            target_policy: the target policy to use for the expert
            expert_probability: probability of sampling from the expert
            render: whether to render the environment
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.render = render > 0
        self.buffer = ReplayBuffer[BaseData](buffer_size)
        self.start_steps = start_steps
        self.automatic_optimization = False
        self.ac_target = deepcopy(self.ac)
    
    def critic_loss(self, centralized_state, action, reward, next_state, next_centralized_state, done, data):
        q = self.ac.critic(centralized_state, action, data)
        
        with torch.no_grad():
            next_mu, next_sigma = self.ac_target.actor(next_state, data)
            next_action = self.ac.policy(next_mu, next_sigma)
            q_target = self.ac_target.critic(next_centralized_state, next_action, data)

            bellman = reward + self.gamma * ~done * q_target
        
        loss = F.mse_loss(q, bellman)

        return loss
    
    def actor_loss(self, state, centralized_state, data):
        mu, sigma = self.ac.actor(state, data)
        action = self.ac.policy(mu, sigma)
        q = self.ac.critic(centralized_state, action, data)
        return -q.mean()

    def training_step(self, data, batch_idx):
        while len(self.buffer) < self.start_steps:
            self.buffer.extend(self.rollout(render=self.render))

        opt_actor, opt_critic = self.optimizers()

        # Update the critic function
        loss_q = self.critic_loss(data.centralized_state, data.action, data.reward, data.next_state, data.next_centralized_state, data.done, data)
        self.log("train/critic_loss", loss_q, prog_bar=True, batch_size=data.batch_size)
        opt_critic.zero_grad()
        self.manual_backward(loss_q)
        opt_critic.step()

        # Freeze the critic network
        for p in self.ac.critic.parameters():
            p.requires_grad = False

        # Update the actor function
        loss_pi = self.actor_loss(data.state, data.centralized_state, data)
        self.log("train/actor_loss", loss_pi, prog_bar=True, batch_size=data.batch_size)
        opt_actor.zero_grad()
        self.manual_backward(loss_pi)
        opt_actor.step()

        # Unfreeze the critic network
        for p in self.ac.critic.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_target in zip(self.ac.parameters(), self.ac_target.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)

    def validation_step(self, data, batch_idx):
        loss_q = self.critic_loss(data.centralized_state, data.action, data.reward, data.next_state, data.next_centralized_state, data.done, data)
        loss_pi = self.actor_loss(data.state, data.centralized_state, data)

        self.log("val/critic_loss", loss_q, prog_bar=True, batch_size=data.batch_size)
        self.log("val/actor_loss", loss_pi, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "val/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )

        return loss_q, loss_pi

    def test_step(self, data, batch_idx):
        loss_q = self.critic_loss(data.centralized_state, data.action, data.reward, data.next_state, data.next_centralized_state, data.done, data)
        loss_pi = self.actor_loss(data.state, data.centralized_state, data)

        self.log("test/critic_loss", loss_q, prog_bar=True, batch_size=data.batch_size)
        self.log("test/actor_loss", loss_pi, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "test/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )

        return loss_q, loss_pi

    def rollout_start(self):
        pass

    def rollout_step(
        self,
        data: BaseData,
    ):
        if self.training:
            # use actor policy
            data.action = self.ac.actor.policy(data.mu, data.sigma)
        else:
            # use greedy policy
            data.action = data.mu

        next_state, centralized_state, reward, done, _ = self.env.step(data.action.detach().cpu().numpy())
        return data, next_state, centralized_state, reward, done

    def batch_generator(
        self, n_episodes=1, render=False, use_buffer=True, training=True
    ):
        # set model to appropriate mode
        self.train(training)

        data = []
        for _ in range(n_episodes):
            data.extend(self.rollout(render=render))
        if use_buffer:
            self.buffer.extend(data)
            data = self.buffer.collect(shuffle=True)
        return iter(data)

    def train_dataloader(self):
        return self._dataloader(
            n_episodes=10, render=False, use_buffer=True, training=True
        )

    def val_dataloader(self):
        return self._dataloader(
            n_episodes=1, render=self.render, use_buffer=False, training=False
        )

    def test_dataloader(self):
        return self._dataloader(
            n_episodes=100, render=self.render, use_buffer=False, training=False
        )
