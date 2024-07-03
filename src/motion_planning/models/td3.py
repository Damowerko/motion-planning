from itertools import chain

import numpy as np

from motion_planning.models.base import *
from motion_planning.rl import ReplayBuffer


class MotionPlanningTD3(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        buffer_size: int = 100_000,
        start_steps: int = 2_000,
        noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
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
        self.noise = noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.delay_count = 0
        self.automatic_optimization = False
        self.ac.critic2 = deepcopy(self.ac.critic)
        self.ac_target = deepcopy(self.ac)

    def configure_optimizers(self):
        return (
            torch.optim.AdamW(
                self.ac.actor.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay
            ),
            torch.optim.AdamW(
                chain(self.ac.critic.parameters(), self.ac.critic2.parameters()),
                lr=self.critic_lr,
                weight_decay=self.weight_decay,
            ),
        )
    
    def critic_loss(self, centralized_state, action, reward, next_state, next_centralized_state, done, data):
        q1 = self.ac.critic(centralized_state, action, data)
        q2 = self.ac.critic2(centralized_state, action, data)
        
        with torch.no_grad():
            next_mu, _ = self.ac_target.actor(next_state, data)
            eps = torch.randn_like(next_mu) * self.noise
            next_action = next_mu + torch.clip(eps, -self.noise_clip, self.noise_clip)
            q1_target = self.ac_target.critic(next_centralized_state, next_action, data)
            q2_target = self.ac_target.critic2(next_centralized_state, next_action, data)

            bellman = reward + self.gamma * ~done * torch.min(q1_target, q2_target)
        
        loss1 = F.mse_loss(q1, bellman)
        loss2 = F.mse_loss(q2, bellman)
        loss = loss1 + loss2

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

        self.delay_count += 1

        if self.delay_count % self.policy_delay == 0:
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

        self.log("val/critic_loss", loss_q, prog_bar=True, batch_size=data.batch_size)
        self.log("val/actor_loss", loss_pi, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "val/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
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
