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
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_ndim = 14
        self.action_ndim = 2
        self.embed_dim = n_channels * n_heads

        self.readin_state_encoder = gnn.MLP(
            [self.state_ndim, 2 * self.embed_dim, self.embed_dim],
            norm="layer_norm",
            dropout=dropout,
        )
        self.readin_pos_encoder = gnn.MLP(
            [2, 2 * self.embed_dim, self.embed_dim], norm="layer_norm", dropout=dropout
        )
        self.readin_pos_decoder = gnn.MLP(
            [2, 2 * self.embed_dim, self.embed_dim], norm="layer_norm", dropout=dropout
        )
        self.readin_targets = gnn.MLP(
            [2, 2 * self.embed_dim, self.embed_dim], norm="layer_norm", dropout=dropout
        )
        self.readout = gnn.MLP(
            [self.embed_dim, 2 * self.embed_dim, self.action_ndim],
            plain_last=True,
            norm="layer_norm",
            dropout=dropout,
        )
        self.transformer = nn.Transformer(
            d_model=self.embed_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, data: Data) -> torch.Tensor:
        batch_size = data.batch_size if isinstance(data, Batch) else 1
        positions = data.positions.reshape(batch_size, -1, 2)
        targets = data.targets.reshape(batch_size, -1, 2)
        state = data.state.reshape(batch_size, -1, self.state_ndim)
        positions_emb = self.readin_pos_encoder(positions)
        targets_emb = self.readin_targets(targets)
        state_emb = self.readin_state_encoder(state)
        src = torch.cat([positions_emb + state_emb, targets_emb], dim=1)
        # tgt = self.readin_pos_decoder(positions)
        tgt = positions_emb + state_emb
        y = self.transformer(src, tgt)
        return self.readout(y).reshape(-1, self.action_ndim)


class TransformerCritic(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_channels: int = 32,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = n_channels * n_heads

        self.readin = gnn.MLP(
            [4, 2 * self.embed_dim, self.embed_dim],
            norm="layer_norm",
            dropout=dropout,
        )
        self.readout = gnn.MLP(
            [self.embed_dim, 2 * self.embed_dim, 1],
            plain_last=True,
            norm="layer_norm",
            dropout=dropout,
        )
        self.transformer = nn.Transformer(
            d_model=self.embed_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            batch_first=True,
            dropout=dropout,
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
        dropout: float = 0.0,
        **kwargs,
    ):
        actor = TransformerActor(
            n_layers,
            n_channels,
            n_heads,
            dropout,
        )
        critic = TransformerCritic(
            n_layers,
            n_channels,
            n_heads,
            dropout,
        )
        super().__init__(actor, critic)
