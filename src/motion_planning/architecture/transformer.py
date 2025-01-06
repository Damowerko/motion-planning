import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Batch, Data

from motion_planning.architecture.base import ActorCritic


class TransformerActor(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_channels: int = 256,
        n_heads: int = 8,
    ):
        super().__init__()
        self.action_ndim = 2

        self.readin_pos_encoder = gnn.MLP(
            [2, 2 * n_channels, n_channels], norm="layer_norm"
        )
        self.readin_pos_decoder = gnn.MLP(
            [2, 2 * n_channels, n_channels], norm="layer_norm"
        )
        self.readin_targets = gnn.MLP(
            [2, 2 * n_channels, n_channels], norm="layer_norm"
        )
        self.readout = gnn.MLP(
            [n_channels, 2 * n_channels, self.action_ndim],
            plain_last=True,
            norm="layer_norm",
        )
        self.transformer = nn.Transformer(
            d_model=n_channels,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            batch_first=True,
        )

    def forward(self, data: Data) -> torch.Tensor:
        batch_size = data.batch_size if isinstance(data, Batch) else 1
        positions = data.positions.reshape(batch_size, -1, 2)
        targets = data.targets.reshape(batch_size, -1, 2)
        src = torch.cat(
            [self.readin_pos_encoder(positions), self.readin_targets(targets)], dim=1
        )
        tgt = self.readin_pos_decoder(positions)
        y = self.transformer(src, tgt)
        return self.readout(y).reshape(-1, self.action_ndim)


class TransformerCritic(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_channels: int = 256,
        n_heads: int = 8,
    ):
        super().__init__()

        self.readin = gnn.MLP(
            [4, 2 * n_channels, n_channels],
            norm="layer_norm",
        )
        self.readout = gnn.MLP(
            [n_channels, 2 * n_channels, 1],
            plain_last=True,
            norm="layer_norm",
        )
        self.transformer = nn.Transformer(
            d_model=n_channels,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            batch_first=True,
        )

    def forward(self, action: torch.Tensor, data: Data) -> torch.Tensor:
        batch_size = data.batch_size if isinstance(data, Batch) else 1
        positions = data.positions
        x = torch.cat([positions, action], dim=-1).view(batch_size, -1, 4)
        x = self.readin(x)
        y = self.transformer(x, x)
        z = self.readout(y).reshape(-1)
        return z


class TransformerActorCritic(ActorCritic):
    def __init__(
        self,
        n_layers: int = 4,
        n_channels: int = 256,
        n_heads: int = 8,
        **kwargs,
    ):
        actor = TransformerActor(
            n_layers,
            n_channels,
            n_heads,
        )
        critic = TransformerCritic(
            n_layers,
            n_channels,
            n_heads,
        )
        super().__init__(actor, critic)
