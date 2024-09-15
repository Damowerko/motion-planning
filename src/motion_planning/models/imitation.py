import random

import numpy as np

from motion_planning.models.base import *
from motion_planning.rl import ReplayBuffer


class MotionPlanningImitation(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        buffer_size: int = 100_000,
        target_policy: str = "c",
        expert_probability: float = 0.5,
        expert_probability_decay: float = 0.99,
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
        self.target_policy = target_policy
        self.render = render > 0
        self.buffer = ReplayBuffer[BaseData](buffer_size)
        self.expert_probability = expert_probability
        self.expert_probability_decay = expert_probability_decay
        self.automatic_optimization = False

    def training_step(self, data, batch_idx):
        opt_actor, opt_critic = self.optimizers()

        # actor step
        mu, _ = self.ac.actor.forward(data.state, data)
        loss_pi = F.mse_loss(mu, data.expert)
        self.log("train/mu_loss", loss_pi, prog_bar=True, batch_size=data.batch_size)
        opt_actor.zero_grad()
        self.manual_backward(loss_pi)
        opt_actor.step()

        # critic step
        # q = self.ac.critic.forward(data.centralized_state, data.action, data)
        # with torch.no_grad():
        #     next_action, _ = self.ac.actor.forward(data.next_state, data)
        #     qprime = self.ac.critic.forward(data.next_centralized_state, next_action, data)
        # loss_q = self.critic_loss(q, qprime, data.reward, data.done)
        # self.log("train/critic_loss", loss_q, prog_bar=True, batch_size=data.batch_size)
        # opt_critic.zero_grad()
        # self.manual_backward(loss_q)
        # opt_critic.step()

    def validation_step(self, data, batch_idx):
        mu, _ = self.ac.actor.forward(data.state, data)
        loss = F.mse_loss(mu, data.expert)
        self.log("val/mu_loss", loss, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "val/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )
        self.log("val/coverage", data.coverage.mean(), batch_size=data.batch_size)
        self.log(
            "val/n_collisions", data.n_collisions.mean(), batch_size=data.batch_size
        )

        return loss

    def test_step(self, data, batch_idx):
        mu, _ = self.ac.actor.forward(data.state, data)
        loss = F.mse_loss(mu, data.expert)
        self.log("test/mu_loss", loss, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "test/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )

        return loss

    def rollout_start(self):
        # decide if expert will be used for rollout
        if self.training:
            expert_probability = (
                self.expert_probability
                * self.expert_probability_decay**self.current_epoch
            )
            self.use_expert = random.random() < expert_probability
        else:
            self.use_expert = False

    def rollout_step(
        self,
        data: BaseData,
    ):
        if self.target_policy == "c":
            expert = self.env.centralized_policy()
        elif self.target_policy == "d0":
            expert = self.env.decentralized_policy(0)
        elif self.target_policy == "d1":
            expert = self.env.decentralized_policy(1)
        else:
            raise ValueError(f"Unknown target policy {self.target_policy}")

        data.expert = torch.as_tensor(expert, dtype=self.dtype, device=self.device)  # type: ignore

        if self.use_expert:
            # use expert policy
            data.action = data.expert
        elif self.training:
            # use actor policy
            data.action = self.ac.policy(data.mu, data.sigma)
        else:
            # use greedy policy
            data.action = data.mu

        next_state, centralized_state, reward, done, _ = self.env.step(
            data.action.detach().cpu().numpy()
        )
        coverage = self.env.coverage()
        n_collisions = self.env.n_collisions(r=self.agent_radius)
        return data, next_state, centralized_state, reward, done, coverage, n_collisions

    def batch_generator(
        self, n_episodes=1, render=False, use_buffer=True, training=True
    ):
        # set model to appropriate mode
        self.train(training)

        data = []
        for _ in range(n_episodes):
            episode, frames = self.rollout(render=render)
            data.extend(episode)
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
