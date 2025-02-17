import typing
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch_geometric.data import Data
from torchcps.utils import add_model_specific_args

from motion_planning.architecture.base import ActorCritic
from motion_planning.lightning.base import ActorCritic, MotionPlanningActorCritic


class MotionPlanningTD3(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: ActorCritic,
        polyak: float = 0.99,
        policy_delay: int = 2,
        warmup_epochs: int = 0,
        **kwargs,
    ):
        """
        Args:
            buffer_size: size of the replay buffer
            target_policy: the target policy to use for the expert
            expert_probability: probability of sampling from the expert
            warmup_epochs: Number of epochs to take before starting to train the actor.
        """
        super().__init__(model, **kwargs)
        self.policy_delay = policy_delay
        self.warmup_epochs = warmup_epochs
        self.automatic_optimization = False
        self.critics = nn.ModuleList([self.model.critic, deepcopy(self.model.critic)])

        # reset the parameters of the critics, so that they are not the same
        def reset_parameters(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

        self.critics.apply(reset_parameters)

        self.critic_targets = typing.cast(
            list[AveragedModel],
            nn.ModuleList(
                [
                    AveragedModel(
                        critic,
                        multi_avg_fn=get_ema_multi_avg_fn(decay=polyak),
                        use_buffers=True,
                    )
                    for critic in self.critics
                ]
            ),
        )
        self.actor_target = AveragedModel(
            self.model.actor,
            multi_avg_fn=get_ema_multi_avg_fn(decay=polyak),
            use_buffers=True,
        )

    def configure_optimizers(self):
        actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        critic_optimizer = torch.optim.AdamW(
            self.critics.parameters(),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        return [actor_optimizer, critic_optimizer]

    def critic_loss(self, data: Data, next_data: Data) -> torch.Tensor:
        # the target policy and critic is are to critique the next action
        # the current critic is trained to satisfy the MSE on the bellman equation
        with torch.no_grad():
            mu_target = self.model.forward_actor(self.actor_target, next_data)
            next_action = self.policy(mu_target)
            # take the minimum of the two criticsas
            qs_target = [
                self.model.forward_critic(critic, next_action, next_data)
                for critic in self.critic_targets
            ]
            # normally there is a (1-done) term here, but we are never done
            N = mu_target.size(0) // data.batch_size
            done = data.done[:, None].expand(data.batch_size, N).reshape(-1)
            y = data.reward + torch.logical_not(done) * self.gamma * torch.minimum(
                *qs_target
            )
        qs = [
            self.model.forward_critic(critic, data.action, data)
            for critic in self.critics
        ]
        # average over the batch, but sum over the critics
        critic_loss = torch.stack([F.mse_loss(q, y) for q in qs], dim=-1).sum(-1)
        return critic_loss

    def actor_loss(self, data: Data) -> torch.Tensor:
        # actor is trained to maximize the critic
        # do not need to sample stochastic policy, since the critic is differentiable
        action = self.model.forward_actor(self.model.actor, data)
        action = self.clip_action(action)
        q = self.model.forward_critic(self.modeel.critic, action, data)
        return -q.mean()

    def training_step(self, data_pair):
        data, next_data = data_pair
        opt_actor, opt_critic = self.optimizers()

        # critic update
        critic_loss = self.critic_loss(data, next_data)
        opt_actor.zero_grad()
        opt_critic.zero_grad()
        self.manual_backward(critic_loss)
        opt_critic.step()
        for critic, critic_target in zip(self.critics, self.critic_targets):
            critic_target.update_parameters(critic)

        # actor update
        actor_loss = self.actor_loss(data)
        if (
            self.global_step + 1
        ) % self.policy_delay != 0 and self.current_epoch > self.warmup_epochs:
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            self.manual_backward(actor_loss)
            opt_actor.step()
            self.actor_target.update_parameters(self.model.actor)

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
