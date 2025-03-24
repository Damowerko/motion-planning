import logging
from typing import Tuple

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictParams
from torchcps.utils import add_model_specific_args
from torchrl.modules import ActorCriticWrapper, OrnsteinUhlenbeckProcessWrapper
from torchrl.objectives import SoftUpdate
from torchrl.objectives import TD3BCLoss as _TD3BCLoss
from torchrl.objectives import TD3Loss as _TD3Loss

from motion_planning.lightning.base import MotionPlanningActorCritic

logger = logging.getLogger(__name__)
from torchrl.objectives.utils import _reduce


class TD3Loss(_TD3Loss):
    """
    Hacked TD3Loss that does not use vmap.
    """

    actor_network: TensorDictModule
    qvalue_network: TensorDictModule
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams

    def _make_vmap(self):
        self._vmap_qvalue_network00 = self.critic_loop

    def critic_loop(self, td, parameters):
        outputs = []
        for i in range(self.num_qvalue_nets):
            with parameters[i].to_module(self.qvalue_network):
                outputs.append(self.qvalue_network(td[i]))
        return torch.stack(outputs, dim=0)


class TD3BCLoss(_TD3BCLoss):
    """
    Hacked TD3Loss that does not use vmap.
    """

    actor_network: TensorDictModule
    qvalue_network: TensorDictModule
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams

    def _make_vmap(self):
        self._vmap_qvalue_network00 = self.critic_loop

    def critic_loop(self, td, parameters):
        outputs = []
        for i in range(self.num_qvalue_nets):
            with parameters[i].to_module(self.qvalue_network):
                outputs.append(self.qvalue_network(td[i]))
        return torch.stack(outputs, dim=0)

    def actor_loss(self, tensordict) -> Tuple[torch.Tensor, dict]:
        """Compute the actor loss.

        The actor loss should be computed after the :meth:`~.qvalue_loss` and is usually delayed 1-3 critic updates.

        Args:
            tensordict (TensorDictBase): the input data for the loss. Check the class's `in_keys` to see what fields
                are required for this to be computed.
        Returns: a differentiable tensor with the actor loss along with a metadata dictionary containing the detached `"bc_loss"`
                used in the combined actor loss as well as the detached `"state_action_value_actor"` used to calculate the lambda
                value, and the lambda value `"lmbd"` itself.
        """
        tensordict_actor_grad = tensordict.select(
            *self.actor_network.in_keys, strict=False
        )
        with self.actor_network_params.to_module(self.actor_network):  # type: ignore
            tensordict_actor_grad = self.actor_network(tensordict_actor_grad)
        actor_loss_td = tensordict_actor_grad.select(
            *self.qvalue_network.in_keys, strict=False
        ).expand(
            self.num_qvalue_nets, *tensordict_actor_grad.batch_size
        )  # for actor loss
        state_action_value_actor = (
            self._vmap_qvalue_network00(
                actor_loss_td,
                self._cached_detach_qvalue_network_params,
            )
            .get(self.tensor_keys.state_action_value)  # type: ignore
            .squeeze(-1)
        )

        bc_loss = torch.nn.functional.mse_loss(
            tensordict_actor_grad.get(self.tensor_keys.action),  # type: ignore
            tensordict.get(self.tensor_keys.action),  # type: ignore
        )
        loss_actor = -state_action_value_actor[0] + self.alpha * bc_loss

        metadata = {
            "state_action_value_actor": state_action_value_actor[0].detach(),
            "bc_loss": bc_loss.detach(),
            "lmbd": 1.0,
        }
        loss_actor = _reduce(loss_actor, reduction=self.reduction)
        return loss_actor, metadata  # type: ignore


class MotionPlanningTD3(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: ActorCriticWrapper,
        polyak: float = 0.995,
        policy_delay: int = 2,
        warmup_epochs: int = 0,
        grad_clip_norm: float = 0.1,
        grad_clip_p: float = 2.0,
        expert_policy: str = "c_sq",
        expert_weight: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            polyak: the polyak value for the target networks
            policy_delay: the delay for the policy update
            warmup_epochs: Number of epochs to take before starting to train the actor.
            grad_clip_norm: the gradient clipping norm
            grad_clip_p: the gradient clipping p
            expert_policy: the expert policy to use
            expert_weight: the weight for the expert policy
        """
        super().__init__(model, **kwargs)
        self.policy_delay = policy_delay
        self.warmup_epochs = warmup_epochs
        self.automatic_optimization = False
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_p = grad_clip_p
        self.exploration_policy = OrnsteinUhlenbeckProcessWrapper(self.model.get_policy_operator())

        if expert_weight > 0.0:
            self.expert_policy = expert_policy
            self.loss = TD3BCLoss(
                self.exploration_policy,
                self.model.get_value_operator(),
                loss_function="l2",
                bounds=(-1, 1),  # type: ignore
                policy_noise=self.noise,
                noise_clip=self.noise_clip,
                # above I monkey patched the TD3BCLoss to not use the lambda value, so alpha is just the weight for the BC loss
                alpha=expert_weight,
            )
            self.loss.make_value_estimator(
                TD3BCLoss.default_value_estimator, gamma=self.gamma
            )
        else:
            self.loss = TD3Loss(
                self.exploration_policy,
                self.model.get_value_operator(),
                loss_function="l2",
                bounds=(-1, 1),  # type: ignore
                policy_noise=self.noise,
                noise_clip=self.noise_clip,
            )
            self.loss.make_value_estimator(
                TD3Loss.default_value_estimator, gamma=self.gamma
            )
        self.target_net_updater = SoftUpdate(self.loss, eps=polyak)

    def rollout_action(self, td: TensorDictBase) -> TensorDictBase:
        """
        Choose an action to take in a rollout step.

        Returns:
            Modified data with data.action set to the action. Can set other fields as well.
        """
        with torch.no_grad():
            td = self.exploration_policy(td)
            if self.training:
                td["action"] = self.policy(td["action"])
            else:
                td["action"] = self.clip_action(td["action"])
            return td

    def populate(self):
        # populate replay buffer with initial data
        logger.info("Populating replay buffer with one rollout.")
        for i, data in enumerate(self.collector):
            if i >= self.max_steps:
                break
            self.buffer.extend(data)
        # add 10 random rollouts
        for _ in range(10):
            data = self.env.rollout(self.max_steps)
            self.buffer.extend(data.reshape(self.num_workers * self.max_steps))  # type: ignore

    def _lr_lambda(self, epoch):
        if epoch < self.warmup_epochs // 2:
            return 0.0
        elif epoch < self.warmup_epochs:
            return (epoch - self.warmup_epochs // 2) / (self.warmup_epochs // 2)
        else:
            return 1.0

    def configure_optimizers(self):
        actor_optimizer = torch.optim.AdamW(
            self.loss.actor_network_params.flatten_keys().values(),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            actor_optimizer, lr_lambda=self._lr_lambda
        )
        critic_optimizer = torch.optim.AdamW(
            self.loss.qvalue_network_params.flatten_keys().values(),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        return [actor_optimizer, critic_optimizer], [actor_scheduler]

    def training_step(self, td: TensorDictBase):
        opt_actor, opt_critic = self.optimizers()

        # TD3BC adds a MSE between "action" and the output of the actor network
        # the action field is not used anywhere else, besides for getting dimensions
        if self.expert_policy is not None:
            td["action"] = td["expert"]
        loss_vals = self.loss(td.clone())
        # actor update
        if (self.global_step + 1) % (self.policy_delay + 1) == 0:
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            self.manual_backward(loss_vals["loss_actor"])
            if self.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.loss.actor_network_params.flatten_keys().values(),
                    max_norm=self.grad_clip_norm,
                    norm_type=self.grad_clip_p,
                )
            opt_actor.step()
        # critic update
        opt_actor.zero_grad()
        opt_critic.zero_grad()
        self.manual_backward(loss_vals["loss_qvalue"])
        if self.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.loss.actor_network_params.flatten_keys().values(),
                max_norm=self.grad_clip_norm,
                norm_type=self.grad_clip_p,
            )
        opt_critic.step()

        self.target_net_updater.step()

        self.log("train/actor_loss", loss_vals["loss_actor"], prog_bar=True)
        self.log("train/critic_loss", loss_vals["loss_qvalue"], prog_bar=True)
        self.log("train/pred_value", loss_vals["pred_value"].mean())
        self.log("train/target_value", loss_vals["target_value"].mean())
        if "bc_loss" in loss_vals:
            self.log("train/bc_loss", loss_vals["bc_loss"].mean())
        # do not log reward, coverage or collisions, since using a replay buffer

    def on_train_epoch_end(self):
        # log the current learning rate
        actor_optimizer, _ = self.optimizers()
        self.log("train/actor_lr", actor_optimizer.param_groups[0]["lr"])
        # update the learning rate
        schedulers = self.lr_schedulers()
        schedulers = schedulers if isinstance(schedulers, list) else [schedulers]
        for scheduler in schedulers:
            if scheduler is not None:
                scheduler.step()  # type: ignore

    def validation_step(self, td: TensorDictBase):
        loss_vals = self.loss(td.clone())
        self.log(
            "val/actor_loss",
            loss_vals["loss_actor"],
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/critic_loss",
            loss_vals["loss_qvalue"],
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/reward",
            td["next", "reward"].mean(),
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/coverage",
            td["coverage"].mean(),
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/collisions",
            td["collisions"].float().mean(),
            batch_size=self.batch_size,
        )

    def test_step(self, td: TensorDictBase):
        loss_vals = self.loss(td.clone())
        self.log(
            "test/actor_loss",
            loss_vals["loss_actor"],
            batch_size=self.batch_size,
        )
        self.log(
            "test/critic_loss",
            loss_vals["loss_qvalue"],
            batch_size=self.batch_size,
        )
        self.log(
            "test/reward",
            td["next", "reward"].mean(),
            batch_size=self.batch_size,
        )
        self.log("test/coverage", td["coverage"].mean(), batch_size=self.batch_size)
        self.log(
            "test/collisions",
            td["collisions"].float().mean(),
            batch_size=self.batch_size,
        )
