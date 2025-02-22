import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torchcps.utils import add_model_specific_args
from torchrl.modules import ActorCriticWrapper
from typing_extensions import override

from motion_planning.lightning.base import MotionPlanningActorCritic


class MotionPlanningImitation(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: ActorCriticWrapper,
        expert_policy: str = "c_sq",
        expert_probability: float = 0.5,
        expert_probability_decay: float = 0.99,
        **kwargs,
    ):
        """
        Args:
            buffer_size: size of the replay buffer
            expert_policy: the target policy to use for the expert
            expert_probability: probability of sampling from the expert
            expert_probability_decay: decay factor for the expert probability
        """
        # these two must go before super().__init__
        super().__init__(model, **kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.expert_policy = expert_policy
        self.expert_probability = expert_probability
        self.expert_probability_decay = expert_probability_decay
        self.automatic_optimization = False

    def setup(self, stage: str):
        self.use_expert = torch.full(
            (self.num_workers,), False, device=self.device, dtype=torch.bool
        )
        super().setup(stage)

    def training_step(self, td: TensorDictBase):
        opt_actor, opt_critic = self.optimizers()
        # actor step
        td_actor = self.model.get_policy_operator()(td.clone())
        loss_actor = F.mse_loss(td_actor["action"], td_actor["expert"])
        self.log("train/actor_loss", loss_actor, prog_bar=True)
        opt_actor.zero_grad()
        self.manual_backward(loss_actor)
        opt_actor.step()

    def validation_step(self, td: TensorDictBase):
        td = self.model.get_policy_operator()(td)
        loss_actor = F.mse_loss(td["action"], td["expert"])
        self.log("val/actor_loss", loss_actor)
        self.log("val/coverage", td["coverage"].mean(), prog_bar=True)
        self.log("val/reward", td["next", "reward"].mean())
        self.log("val/collisions", td["collisions"].float().mean())

    def rollout_start(self):
        # decide if expert will be used for rollout
        with torch.no_grad():
            if self.training:
                expert_probability = (
                    self.expert_probability
                    * self.expert_probability_decay**self.current_epoch
                )
                self.use_expert = (
                    torch.rand((self.num_workers,), device="cuda") < expert_probability
                )
            else:
                self.use_expert = torch.full(
                    (self.num_workers,), False, device="cuda", dtype=torch.bool
                )

    @override
    def rollout_action(self, td: TensorDictBase) -> TensorDictBase:
        with torch.no_grad():
            if self.use_expert.all():
                # use expert policy
                td["action"] = td["expert"]
            td["action"] = torch.where(
                self.use_expert[:, None, None],
                td["expert"],
                self.model.get_policy_operator()(td)["action"],
            )
            return td
