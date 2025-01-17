import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Batch, Data

from motion_planning.architecture.base import ActorCritic

logger = logging.getLogger(__name__)


def linear_frequencies(period: float, n_frequencies: int):
    return 2 * torch.pi * torch.arange(1, 1 + n_frequencies) / period


def geometric_frequencies(period: float, n_frequencies: int):
    # geometric frequency, multiply frequency by (n_frequencies - 1) / n_frequencies
    exponent = torch.arange(n_frequencies, 0, -1) / n_frequencies
    return 2 * torch.pi * torch.pow(1 / period, exponent)


def generate_frequencies(
    period: float, n_frequencies: int, frequency_generator: str = "linear"
):
    if frequency_generator == "linear":
        return linear_frequencies(period, n_frequencies)
    elif frequency_generator == "geometric":
        return geometric_frequencies(period, n_frequencies)
    else:
        raise ValueError(f"Unknown frequency generator: {frequency_generator}")


class AbsolutePositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        period: float,
        n_dimensions: int = 2,
        frequency_generator: str = "linear",
    ):
        """
        Absolute positional encoding layer. Each position is represented by a complex exponential.

        Args:
            embed_dim: The dimensionality of the embedding.
            period: The period of the embedding. Attention will have this period.
            n_dimensions: The number of dimensions of the input data.
            frequency_generator: The method to generate the frequencies. Either "linear" or "geometric".

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.period = period
        self.n_dimensions = n_dimensions
        if embed_dim % (n_dimensions * 2) != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} must be divisible by 2 * n_dimensions"
            )
        self.n_frequencies = self.embed_dim // self.n_dimensions // 2
        self.register_buffer(
            "frequencies",
            generate_frequencies(period, self.n_frequencies, frequency_generator),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: The positions of the input tensor with shape (..., n_dimensions).
        """
        # broadcast so that the angles have shape (..., self.n_frequencies, n_dimensions)
        angle = pos[..., None] * self.frequencies[None, :]
        # stack cos and sin terms along the 2nd to last dimension
        # embedding will have shape (..., n_dimensions, self.embed_dim, 2)
        embedding = torch.stack((angle.cos(), angle.sin()), dim=-1)
        # combine the last three dimensions to get the final embedding
        embedding = embedding.reshape(*pos.shape[:-1], self.embed_dim)
        return embedding


class RotaryPositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        period: float,
        n_dimensions: int = 2,
        frequency_generator: str = "linear",
    ):
        """
        Rotary positional encoding layer. Each position is represented by a complex exponential.

        Args:
            embed_dim: The dimensionality of the embedding.
            period: The period of the embedding.
            n_dimensions: The number of dimensions of the input data.
            frequency_generator: The method to generate the frequencies. Either "linear" or "geometric".

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.period = period
        self.n_dimensions = n_dimensions
        if embed_dim % (n_dimensions * 2) != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} must be divisible by 2 * n_dimensions"
            )
        # number of frequencies per dimension, per complex number
        self.n_frequencies = self.embed_dim // self.n_dimensions // 2
        self.register_buffer(
            "frequencies",
            generate_frequencies(period, self.n_frequencies, frequency_generator),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (..., embed_dim). This should be either the query or key tensor.
            pos: The positions of the input tensor with shape (..., n_dimensions).
        """
        # add dimension to pos so that it has shape (..., n_dimensions, 1)
        # making it broadcastable with self.frequencies of shape (1, n_frequencies)
        angle = pos[..., None] * self.frequencies[None, :]
        # repeat the angle for each complex number in the embedding
        # angle = [theta_1, theta_1, theta_2, theta_2, ...]
        # angle will now have shape (..., n_dimensions, 2 * n_frequencies)
        angle = angle.repeat_interleave(2, dim=-1)
        # we want to arange angle so that the first [2*n_frequencies] are for the first dimension
        # the next [2*n_frequencies] are for the second dimension, etc.
        # I use the fact that torch arrays are row-major, do last index changes first
        angle = angle.reshape(x.shape)
        # reshape, then swap, negate and flatten to get [-x_2, x_1, -x_4, x_3, -x_6, x_5, ...]
        x_swapped = x.reshape(*x.shape[:-1], self.embed_dim // 2, 2).flip(-1)
        x_swapped[..., 0] = x_swapped[..., 0].neg()
        x_swapped = x_swapped.reshape(x.shape)
        # y_1 = x_1 * cos(theta_1) - x_2 * sin(theta_1)
        # y_2 = x_2 * sin(theta_1) + x_1 * cos(theta_1)
        embedding = angle.cos() * x + angle.sin() * x_swapped
        embedding = embedding.reshape(x.shape)
        return embedding


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        encoding_type: str | None = None,
        period: float = 10.0,
        frequency_generator: str = "linear",
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads

        # will have n_layers for encoder and n_layers for decoder
        self.Wq = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )
        self.Wk = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )
        self.Wv = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )
        self.Wo = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )

        self.norm1 = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(n_layers)]
        )
        self.norm2 = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(n_layers)]
        )
        self.linear1 = nn.ModuleList(
            [nn.Linear(self.embed_dim, 2 * self.embed_dim) for _ in range(n_layers)]
        )
        self.linear2 = nn.ModuleList(
            [nn.Linear(2 * self.embed_dim, self.embed_dim) for _ in range(n_layers)]
        )

        if encoding_type is None:
            self.encoding = None
        elif encoding_type == "absolute":
            self.encoding = AbsolutePositionalEncoding(
                self.embed_dim, period, 2, frequency_generator
            )
        elif encoding_type == "rotary":
            self.encoding = RotaryPositionalEncoding(
                self.embed_dim, period, 2, frequency_generator
            )
        elif encoding_type == "mlp":
            self.encoding = gnn.MLP(
                [2, 2 * self.embed_dim, self.embed_dim],
                norm="layer_norm",
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown positional encoding: {encoding_type}")

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B, N = x.shape[:2]
        if isinstance(self.encoding, (AbsolutePositionalEncoding, gnn.MLP)):
            x = x + self.encoding(pos)

        for i in range(self.n_layers):
            # using pre-norm transformer, so we apply layer norm before attention
            x_norm = self.norm1[i](x)
            q = self.Wq[i](x_norm)
            k = self.Wk[i](x_norm)
            v = self.Wv[i](x_norm)
            # rotary positional encoding needs to be applied to the query and keys
            if i == 0 and isinstance(self.encoding, RotaryPositionalEncoding):
                q = self.encoding(q, pos)
                k = self.encoding(k, pos)
            # need to reshape k,q,v to have shape (B, n_heads, N, head_dim)
            # right now we have (B, N, n_heads * head_dim)
            q = q.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

            # compute attention and reshape to "concate" heads
            x_attention = (
                F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
                .transpose(1, 2)
                .reshape(B, N, self.embed_dim)
            )
            x_attention = self.Wo[i](x_attention)
            x = x + x_attention
            # compute the feed forward layer and apply residual connection, as before
            x_fc = self.norm1[i](x_attention)
            x_fc = self.linear1[i](x_fc).relu()
            x_fc = self.linear2[i](x_fc)
            x = x + x_fc
        return x


class TransformerActor(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_channels: int = 32,
        n_heads: int = 8,
        dropout: float = 0.0,
        encoding_type: str = "mlp",
    ):
        super().__init__()
        self.state_ndim = 14
        self.action_ndim = 2
        self.embed_dim = n_channels * n_heads

        self.readin = gnn.MLP(
            [self.state_ndim, 2 * self.embed_dim, self.embed_dim],
            norm="layer_norm",
            dropout=dropout,
        )
        self.readout = gnn.MLP(
            [self.embed_dim, 2 * self.embed_dim, self.action_ndim],
            plain_last=True,
            norm="layer_norm",
            dropout=dropout,
        )
        self.transformer = Transformer(
            n_layers, self.embed_dim, n_heads, dropout, encoding_type=encoding_type
        )

    def forward(self, data: Data) -> torch.Tensor:
        batch_size = data.batch_size if isinstance(data, Batch) else 1
        state = data.state.reshape(batch_size, -1, self.state_ndim)
        x = self.readin(state)
        pos = data.positions.reshape(batch_size, -1, 2)
        y = self.transformer(x, pos)
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
        x = torch.cat([positions, action], dim=-1).reshape(batch_size, -1, 4)
        x = self.readin(x)
        y = self.transformer(x, x)
        z = self.readout(y).reshape(-1)
        return z


class TransformerActorCritic(ActorCritic):
    def __init__(
        self,
        n_layers: int = 2,
        n_channels: int = 32,
        n_heads: int = 2,
        dropout: float = 0.0,
        encoding_type: str = "mlp",
        **kwargs,
    ):
        actor = TransformerActor(
            n_layers,
            n_channels,
            n_heads,
            dropout,
            encoding_type=encoding_type,
        )
        critic = TransformerCritic(
            n_layers,
            n_channels,
            n_heads,
            dropout,
        )
        super().__init__(actor, critic, **kwargs)
