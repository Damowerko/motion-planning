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
        self.actor = actor
        self.critic = critic

        self.actor = torch.compile(self.actor, dynamic=False, disable=not compile)
        self.critic = torch.compile(self.critic, dynamic=False, disable=not compile)

    def forward_actor(self, data: Data) -> torch.Tensor:
        """
        Returns normalized action within the range [-1, 1].
        """
        return self.actor(data).tanh()

    def forward_critic(self, action: torch.Tensor, data: Data) -> torch.Tensor:
        return self.critic(action, data)

    def state_dict(self, *args, **kwargs):
        if not hasattr(self.actor, "_orig_mod") or not hasattr(
            self.critic, "_orig_mod"
        ):
            actor_state_dict = self.actor._orig_mod.state_dict(*args, **kwargs)
            critic_state_dict = self.critic._orig_mod.state_dict(*args, **kwargs)
        else:
            actor_state_dict = self.actor.state_dict(*args, **kwargs)
            critic_state_dict = self.critic.state_dict(*args, **kwargs)
        return {
            "actor": actor_state_dict,
            "critic": critic_state_dict,
        }
