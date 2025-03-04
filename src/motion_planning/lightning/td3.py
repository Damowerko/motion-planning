import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictParams
from torchcps.utils import add_model_specific_args
from torchrl.modules import ActorCriticWrapper
from torchrl.objectives import SoftUpdate
from torchrl.objectives import TD3Loss as _TD3Loss

from motion_planning.lightning.base import MotionPlanningActorCritic


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


class MotionPlanningTD3(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: ActorCriticWrapper,
        polyak: float = 0.99,
        policy_delay: int = 1,
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
        self.loss = TD3Loss(
            self.model.get_policy_operator(),
            self.model.get_value_operator(),
            loss_function="smooth_l1",
            bounds=(-1, 1),  # type: ignore
            policy_noise=self.noise,
            noise_clip=self.noise_clip,
        )
        self.loss.make_value_estimator(
            TD3Loss.default_value_estimator, gamma=self.gamma
        )
        self.target_net_updater = SoftUpdate(self.loss, eps=polyak)

    def configure_optimizers(self):
        actor_optimizer = torch.optim.AdamW(
            self.loss.actor_network_params.flatten_keys().values(),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        critic_optimizer = torch.optim.AdamW(
            self.loss.qvalue_network_params.flatten_keys().values(),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        return [actor_optimizer, critic_optimizer]

    def training_step(self, td: TensorDictBase):
        opt_actor, opt_critic = self.optimizers()
        loss_vals = self.loss(td.clone())
        # actor update
        if (
            self.global_step + 1
        ) % self.policy_delay == 0 and self.current_epoch >= self.warmup_epochs:
            opt_actor.zero_grad()
            self.manual_backward(loss_vals["loss_actor"])
            opt_actor.step()
        # critic update
        opt_critic.zero_grad()
        self.manual_backward(loss_vals["loss_qvalue"])
        opt_critic.step()

        self.target_net_updater.step()

        self.log("train/actor_loss", loss_vals["loss_actor"], prog_bar=True)
        self.log("train/critic_loss", loss_vals["loss_qvalue"], prog_bar=True)
        self.log("train/pred_value", loss_vals["pred_value"].mean())
        self.log("train/target_value", loss_vals["target_value"].mean())
        # do not log reward, coverage or collisions, since using a replay buffer

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
