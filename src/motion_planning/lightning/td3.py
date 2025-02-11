from copy import deepcopy
from pathlib import Path
from typing import Optional

import imageio.v3 as iio

from motion_planning.lightning.base import *
from motion_planning.rl import ReplayBuffer


def save_results(name: str, path: Path, frames: np.ndarray):
    """
    Args:
        path (Path): The path to save the summary to.
        rewards (np.ndarray): An ndarray of shape (n_trials, max_steps).
        frames (np.ndarray): An array of shape (n_trial, max_steps, H, W).
    """
    path.mkdir(parents=True, exist_ok=True)

    # make a single video of all trials
    iio.imwrite(path / f"{name}.mp4", np.concatenate(frames, axis=0), fps=30)


class MotionPlanningTD3(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        buffer_size: int = 100_000,
        start_steps: int = 2_000,
        early_stopping: int = 50,
        policy_delay: int = 2,
        pretrain: bool = False,
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
        self.render = render > 0
        self.buffer = ReplayBuffer[Data](buffer_size)
        self.start_steps = start_steps
        self.early_stopping = early_stopping
        self.policy_delay = policy_delay
        self.delay_count = 0
        self.automatic_optimization = False
        self.model.critic2 = deepcopy(self.model.critic)
        self.ac_target = deepcopy(self.model)

        if pretrain:
            self.frozen_epochs = 100
            self.warmup_epochs = 50
        else:
            self.frozen_epochs = 0
            self.warmup_epochs = 0

    def configure_optimizers(self):
        return (
            torch.optim.AdamW(
                self.model.actor.parameters(),
                lr=self.actor_lr,
                weight_decay=self.weight_decay,
            ),
            torch.optim.AdamW(
                self.model.critic.parameters(),
                lr=self.critic_lr,
                weight_decay=self.weight_decay,
            ),
            torch.optim.AdamW(
                self.model.critic2.parameters(),
                lr=self.critic_lr,
                weight_decay=self.weight_decay,
            ),
        )

    def policy(
        self,
        mu: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        """
        Sample from a Gaussian distribution.

        Args:
            mu: (batch_size, N) mean of the Gaussian distribution
            sigma: (batch_size, N) standard deviation of the Gaussian distribution.
            action: (batch_size, N) Optional action. If given returns tuple action, log_prob, entropy.

        Returns:
            action: (batch_size, N) sample action or given action.
        """
        if action is None:
            eps = torch.randn_like(mu)
            action = mu + sigma * eps
            assert isinstance(action, torch.Tensor)
            return action

        dist = torch.distributions.Normal(mu, sigma)
        return action, dist.log_prob(action), dist.entropy()

    def update_optimizers(self):
        opt_actor, opt_critic1, opt_critic2 = self.optimizers()
        if self.current_epoch < self.frozen_epochs:
            for g in opt_actor.param_groups:
                g["lr"] = 0
            for g in opt_critic1.param_groups:
                g["lr"] = 2 * self.critic_lr
            for g in opt_critic2.param_groups:
                g["lr"] = 2 * self.critic_lr
        elif self.current_epoch < self.frozen_epochs + self.warmup_epochs:
            warmed_rate = (self.current_epoch - self.frozen_epochs) / self.warmup_epochs
            for g in opt_actor.param_groups:
                g["lr"] = warmed_rate * self.actor_lr
            for g in opt_critic1.param_groups:
                g["lr"] = (2 - warmed_rate) * self.critic_lr
            for g in opt_critic2.param_groups:
                g["lr"] = (2 - warmed_rate) * self.critic_lr
        else:
            for g in opt_actor.param_groups:
                g["lr"] = self.actor_lr
            for g in opt_critic1.param_groups:
                g["lr"] = self.critic_lr
            for g in opt_critic2.param_groups:
                g["lr"] = self.critic_lr

    def critic_loss(
        self,
        state,
        positions,
        targets,
        action,
        reward,
        next_state,
        next_positions,
        next_targets,
        done,
        data,
    ):
        # q1 = self.ac.critic(positions, targets, action, data)
        # q2 = self.ac.critic2(positions, targets, action, data)
        q1 = self.model.critic(state, action, data)
        q2 = self.model.critic2(state, action, data)  # type: ignore

        with torch.no_grad():
            next_mu, _ = self.ac_target.actor(next_state, data)
            eps = torch.randn_like(next_mu) * self.noise
            next_action = next_mu + torch.clip(eps, -self.noise_clip, self.noise_clip)
            next_action = self.clip_action(next_action)
            # q1_target = self.ac_target.critic(next_positions, targets, next_action, data)
            # q2_target = self.ac_target.critic2(next_positions, targets, next_action, data)
            q1_target = self.ac_target.critic(next_state, next_action, data)
            q2_target = self.ac_target.critic2(next_state, next_action, data)  # type: ignore

            # bellman = reward + self.gamma * ~done * torch.min(q1_target, q2_target)
            bellman = reward + self.gamma * torch.min(q1_target, q2_target)

        self.log("vals/q1", q1.mean(), batch_size=data.batch_size)
        self.log("vals/q2", q2.mean(), batch_size=data.batch_size)
        self.log("vals/reward", reward.mean(), batch_size=data.batch_size)
        self.log("vals/bellman", bellman.mean(), batch_size=data.batch_size)

        loss1 = F.mse_loss(q1, bellman)
        loss2 = F.mse_loss(q2, bellman)

        return loss1, loss2

    def actor_loss(self, data):
        action, _ = self.model.forward_actor(data)
        self.log(
            "vals/action_magnitude",
            torch.norm(action, dim=-1).mean(),
            prog_bar=True,
            batch_size=data.batch_size,
        )
        action = self.clip_action(action)
        # q = self.ac.critic(positions, targets, action, data)
        q = self.model.critic(state, action, data)
        return -q.mean()

    def training_step(self, data, batch_idx):
        self.update_optimizers()
        opt_actor, opt_critic1, opt_critic2 = self.optimizers()

        # Update the critic function
        loss_q1, loss_q2 = self.critic_loss(
            data.state,
            data.positions,
            data.targets,
            data.action,
            data.reward,
            data.next_state,
            data.next_positions,
            data.next_targets,
            data.done,
            data,
        )
        self.log(
            "train/critic1_loss", loss_q1, prog_bar=True, batch_size=data.batch_size
        )
        self.log(
            "train/critic2_loss", loss_q2, prog_bar=True, batch_size=data.batch_size
        )

        opt_critic1.zero_grad()
        self.manual_backward(loss_q1)
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 1e-2)
        for name, param in self.model.critic.named_parameters():
            self.log(
                f"train/critic1_gradients/{name}",
                param.grad.mean(),
                batch_size=data.batch_size,
            )
        opt_critic1.step()

        opt_critic2.zero_grad()
        self.manual_backward(loss_q2)
        torch.nn.utils.clip_grad_norm_(self.model.critic2.parameters(), 1e-2)
        for name, param in self.model.critic2.named_parameters():
            self.log(
                f"train/critic2_gradients/{name}",
                param.grad.mean(),
                batch_size=data.batch_size,
            )
        opt_critic2.step()

        self.delay_count += 1

        if (
            self.delay_count % self.policy_delay == 0
            and self.current_epoch >= self.frozen_epochs
        ):
            # Freeze the critic network
            for p in self.model.critic.parameters():
                p.requires_grad = False

            # Update the actor function
            loss_pi = self.actor_loss(data.state, data.positions, targets, data)
            self.log(
                "train/actor_loss", loss_pi, prog_bar=True, batch_size=data.batch_size
            )
            opt_actor.zero_grad()
            self.manual_backward(loss_pi)
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), 1e-2)
            for name, param in self.model.actor.named_parameters():
                if param.grad is None:
                    continue
                self.log(
                    f"train/actor_gradients/{name}",
                    param.grad.mean(),
                    batch_size=data.batch_size,
                )
            opt_actor.step()

            # Unfreeze the critic network
            for p in self.model.critic.parameters():
                p.requires_grad = True

        # Perform polyak averaging to update the target actor-critic
        with torch.no_grad():
            for p, p_target in zip(
                self.model.parameters(), self.ac_target.parameters()
            ):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)

    def validation_step(self, data, batch_idx):
        loss_q = self.critic_loss(
            data.state,
            data.positions,
            targets,
            data.action,
            data.reward,
            data.next_state,
            data.next_positions,
            targets,
            data.done,
            data,
        )
        loss_pi = self.actor_loss(data.state, data.positions, targets, data)

        self.log(
            "val/critic1_loss", loss_q[0], prog_bar=True, batch_size=data.batch_size
        )
        self.log(
            "val/critic2_loss", loss_q[1], prog_bar=True, batch_size=data.batch_size
        )
        self.log("val/actor_loss", loss_pi, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "val/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )
        self.log("val/coverage", data.coverage.mean(), batch_size=data.batch_size)
        self.log(
            "val/n_collisions", data.n_collisions.mean(), batch_size=data.batch_size
        )

        return loss_q, loss_pi

    def test_step(self, data, batch_idx):
        loss_q = self.critic_loss(
            data.state,
            data.positions,
            targets,
            data.action,
            data.reward,
            data.next_state,
            data.next_positions,
            targets,
            data.done,
            data,
        )
        loss_pi = self.actor_loss(data.state, data.positions, targets, data)

        self.log("test/critic1_loss", loss_q[0], batch_size=data.batch_size)
        self.log("test/critic2_loss", loss_q[1], batch_size=data.batch_size)
        self.log("test/actor_loss", loss_pi, batch_size=data.batch_size)
        self.log("test/reward", data.reward.mean(), batch_size=data.batch_size)

        return loss_q, loss_pi

    def rollout_action(self, data: Data):
        if self.training:
            # use actor policy
            eps = torch.randn_like(data.mu) * self.noise
            data.action = data.mu + torch.clip(eps, -self.noise_clip, self.noise_clip)
            data.action = self.clip_action(data.action)
        else:
            # use greedy policy
            data.action = data.mu
        return data

    def batch_generator(
        self, n_episodes=1, render=False, use_buffer=True, training=True
    ):
        # set model to appropriate mode
        self.train(training)

        data = []
        for _ in range(n_episodes):
            episode, frames = self.rollout(render=render)
            data.extend(episode)
            # if render and self.current_epoch % 10 == 0:
            #     save_results(f"{self.current_epoch}", Path("figures") / "test_results", frames)
            #     wandb.save(f"figures/test_results/{self.current_epoch}.mp4")
        if use_buffer:
            self.buffer.extend(data)
            data = self.buffer.collect(shuffle=True)
            if training:
                while len(self.buffer) < self.start_steps:
                    episode, frames = self.rollout(render=render)
                    self.buffer.extend(episode)
        return iter(data)

    def train_dataloader(self):
        return self._dataloader(
            n_episodes=10, render=False, use_buffer=True, training=True
        )

    def val_dataloader(self):
        return self._dataloader(
            n_episodes=1, render=True, use_buffer=False, training=False
        )

    def test_dataloader(self):
        return self._dataloader(
            n_episodes=100, render=self.render, use_buffer=False, training=False
        )
