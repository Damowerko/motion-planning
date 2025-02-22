import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchcps.utils import add_model_specific_args
from torchrl.modules import ActorCriticWrapper


class ActorCritic(ActorCriticWrapper):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        actor: TensorDictModule,
        critic: TensorDictModule,
        compile: bool = True,
        **_,
    ):
        """
        Args:
            actor: Actor network, no activation function at last layer.
            critic: Critic network, no activation function at last layer.
            compile: Weather to compile the models.
        """
        super().__init__(actor, critic)
        self.actor: TensorDictModule = actor
        self.critic: TensorDictModule = critic
        self.actor = torch.compile(self.actor, dynamic=False, disable=not compile)  # type: ignore
        self.critic = torch.compile(self.critic, dynamic=False, disable=not compile)  # type: ignore
