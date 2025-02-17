import torch
import torch.nn as nn
from torch_geometric.data import Data
from torchcps.utils import add_model_specific_args


class ActorCritic(nn.Module):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        compile: bool = True,
        **_,
    ):
        """
        Args:
            actor: Actor network, no activation function at last layer.
            critic: Critic network, no activation function at last layer.
            compile: Weather to compile the models.
        """
        super().__init__()
        self.actor: nn.Module = actor
        self.critic: nn.Module = critic

        self.actor = torch.compile(self.actor, dynamic=False, disable=not compile)  # type: ignore
        self.critic = torch.compile(self.critic, dynamic=False, disable=not compile)  # type: ignore

    @staticmethod
    def forward_actor(actor: nn.Module, data: Data) -> torch.Tensor:
        """
        Returns normalized action within the range [-1, 1].
        """
        return actor(data).tanh()

    @staticmethod
    def forward_critic(
        critic: nn.Module, action: torch.Tensor, data: Data
    ) -> torch.Tensor:
        return critic(action, data)
