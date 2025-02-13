from copy import deepcopy
from itertools import chain

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch_geometric.data import Data
from torchcps.utils import add_model_specific_args

from motion_planning.architecture.base import ActorCritic, forward_actor, forward_critic
from motion_planning.lightning.base import ActorCritic, MotionPlanningActorCritic


class MotionPlanningTD3(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: ActorCritic,
        polyak: float = 0.995,
        policy_delay: int = 2,
        warmup_epochs: int = 0,
        **kwargs,
    ):
        """
        Args:
            buffer_size: size of the replay buffer
            target_policy: the target policy to use for the expert
            expert_probability: probability of sampling from the expert
            warmup_epochs: Number of epochs to take before fully training the actor.
                For first `warmup_epochs//2` will train the critic only.
                Then for the next `warmup_epochs//2` will linearly increase the actor learning rate.
        """
        super().__init__(model, **kwargs)
        self.policy_delay = policy_delay
        self.warmup_epochs = warmup_epochs
        self.automatic_optimization = False
        self.critics = nn.ModuleList([self.model.critic, deepcopy(self.model.critic)])
        self.critic_targets = nn.ModuleList(
            [
                AveragedModel(critic, multi_avg_fn=get_ema_multi_avg_fn(decay=polyak))
                for critic in self.critics
            ]
        )
        self.actor_target = AveragedModel(
            self.model.actor, multi_avg_fn=get_ema_multi_avg_fn(decay=polyak)
        )

    def configure_optimizers(self):
        actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        critic_optimizer = torch.optim.AdamW(
            chain(*[critic.parameters() for critic in self.critics]),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        actor_scheduler = torch.optim.lr_scheduler.SequentialLR(
            actor_optimizer,
            [
                torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _: 0.0),
                torch.optim.lr_scheduler.LinearLR(
                    actor_optimizer, 0.01, 1.0, self.warmup_epochs // 2
                ),
            ],
            [self.warmup_epochs // 2],
        )

        return [actor_optimizer, critic_optimizer], [actor_scheduler]

    def critic_loss(self, data: Data, next_data: Data) -> torch.Tensor:
        # the target policy and critic is are to critique the next action
        # the current critic is trained to satisfy the MSE on the bellman equation
        with torch.no_grad():
            mu_target = forward_actor(self.actor_target, next_data)
            next_action = self.policy(mu_target)
            # take the minimum of the two critics
            q_target = torch.minimum(
                *(
                    forward_critic(critic, next_action, next_data)
                    for critic in self.critic_targets
                )
            )
            # normally there is a (1-done) term here, but we are never done
            y = data.reward + self.gamma * q_target
        qs = [forward_critic(critic, data.action, data) for critic in self.critics]
        # average over the batch, but sum over the critics
        critic_loss = torch.stack(
            [F.mse_loss(q, y, reduction="mean") for q in qs], dim=-1
        ).sum(-1)
        return critic_loss

    def actor_loss(self, data: Data) -> torch.Tensor:
        # actor is trained to maximize the critic
        # do not need to sample stochastic policy, since the critic is differentiable
        action = self.model.forward_actor(data)
        action = self.clip_action(action)
        q = self.model.forward_critic(action, data)
        return -q.mean()

    def training_step(self, data_pair):
        data, next_data = data_pair
        opt_actor, opt_critic = self.optimizers()

        # critic update
        critic_loss = self.critic_loss(data, next_data)
        opt_critic.zero_grad()
        self.manual_backward(critic_loss)
        opt_critic.step()

        # actor update
        actor_loss = self.actor_loss(data)
        if (self.global_step + 1) % self.policy_delay != 0:
            opt_actor.zero_grad()
            self.manual_backward(actor_loss)
            opt_actor.step()

        self.lr_schedulers().step()  # type: ignore

        self.log(
            "train/critic_loss", critic_loss, prog_bar=True, batch_size=data.batch_size
        )
        self.log(
            "train/actor_loss", actor_loss, prog_bar=True, batch_size=data.batch_size
        )
        # do not log reward, coverage or n_collisions, since using a replay buffer

    def validation_step(self, data_pair):
        data, next_data = data_pair
        actor_loss = self.actor_loss(data)
        critic_loss = self.critic_loss(data, next_data)
        self.log(
            "val/critic_loss", critic_loss, prog_bar=True, batch_size=data.batch_size
        )
        self.log(
            "val/actor_loss", actor_loss, prog_bar=True, batch_size=data.batch_size
        )
        self.log(
            "val/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )
        self.log("val/coverage", data.coverage.mean(), batch_size=data.batch_size)
        self.log(
            "val/n_collisions", data.n_collisions.mean(), batch_size=data.batch_size
        )

    def test_step(self, data_pair):
        data, next_data = data_pair
        actor_loss = self.actor_loss(data)
        critic_loss = self.critic_loss(data, next_data)
        self.log("test/critic_loss", critic_loss, batch_size=data.batch_size)
        self.log("test/actor_loss", actor_loss, batch_size=data.batch_size)
        self.log("test/reward", data.reward.mean(), batch_size=data.batch_size)
        self.log("test/coverage", data.coverage.mean(), batch_size=data.batch_size)
        self.log(
            "test/n_collisions", data.n_collisions.mean(), batch_size=data.batch_size
        )
