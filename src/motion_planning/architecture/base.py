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
    ):
        """
        Args:
            actor: Actor network, no activation function at last layer.
            critic: Critic network, no activation function at last layer.
        """
        super().__init__()
        self.actor = actor
        self.critic = critic

    @torch.compile(dynamic=False)
    def forward_actor(self, data: Data) -> torch.Tensor:
        """
        Returns normalized action within the range [-1, 1].
        """
        return self.actor(data).tanh()

    @torch.compile(dynamic=False)
    def forward_critic(self, action: torch.Tensor, data: Data) -> torch.Tensor:
        return self.critic(action, data)
