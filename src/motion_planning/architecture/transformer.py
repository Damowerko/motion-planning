import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Batch, Data

from motion_planning.architecture.base import ActorCritic

logger = logging.getLogger(__name__)


def linear_frequencies(encoding_period: float, n_frequencies: int):
    return 2 * torch.pi * torch.arange(1, 1 + n_frequencies) / encoding_period


def geometric_frequencies(encoding_period: float, n_frequencies: int):
    # geometric frequency, multiply frequency by (n_frequencies - 1) / n_frequencies
    exponent = torch.arange(n_frequencies, 0, -1) / n_frequencies
    return 2 * torch.pi * torch.pow(1 / encoding_period, exponent)


def generate_frequencies(
    encoding_period: float, n_frequencies: int, encoding_frequencies: str = "linear"
):
    if encoding_frequencies == "linear":
        return linear_frequencies(encoding_period, n_frequencies)
    elif encoding_frequencies == "geometric":
        return geometric_frequencies(encoding_period, n_frequencies)
    else:
        raise ValueError(f"Unknown frequency generator: {encoding_frequencies}")


class AbsolutePositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        encoding_period: float,
        n_dimensions: int = 2,
        encoding_frequencies: str = "linear",
    ):
        """
        Absolute positional encoding layer. Each position is represented by a complex exponential.

        Args:
            embed_dim: The dimensionality of the embedding.
            encoding_period: The period of the embedding. Attention will have this period.
            n_dimensions: The number of dimensions of the input data.
            encoding_frequencies: The method to generate the frequencies. Either "linear" or "geometric".

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding_period = encoding_period
        self.n_dimensions = n_dimensions
        if embed_dim % (n_dimensions * 2) != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} must be divisible by 2 * n_dimensions"
            )
        self.n_frequencies = self.embed_dim // self.n_dimensions // 2
        self.register_buffer(
            "frequencies",
            generate_frequencies(
                encoding_period, self.n_frequencies, encoding_frequencies
            ),
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
        encoding_period: float,
        n_dimensions: int = 2,
        encoding_frequencies: str = "linear",
    ):
        """
        Rotary positional encoding layer. Each position is represented by a complex exponential.

        Args:
            embed_dim: The dimensionality of the embedding.
            encoding_period: The period of the embedding.
            n_dimensions: The number of dimensions of the input data.
            encoding_frequencies: The method to generate the frequencies. Either "linear" or "geometric".

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding_period = encoding_period
        self.n_dimensions = n_dimensions
        if embed_dim % (n_dimensions * 2) != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} must be divisible by 2 * n_dimensions"
            )
        # number of frequencies per dimension, per complex number
        self.n_frequencies = self.embed_dim // self.n_dimensions // 2
        self.register_buffer(
            "frequencies",
            generate_frequencies(
                encoding_period, self.n_frequencies, encoding_frequencies
            ),
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
        encoding_period: float = 1000.0,
        encoding_frequencies: str = "linear",
        attention_window: float = 0.0,
    ):
        """
        Args:
            n_layers: The number of transformer layers.
            embed_dim: The dimensionality of the embedding. Must be divisible by n_heads.
            n_heads: The number of attention heads. Each head will have dimension embed_dim // n_heads.
            dropout: The dropout rate.
            encoding_type: The type of positional encoding to use. Either "absolute", "rotary", "mlp" or None.
            encoding_period: The period of the encoding. Only used for "absolute" and "rotary" encoding.
            encoding_frequencies: The method to generate the frequencies. Either "linear" or "geometric". Only used for "absolute" and "rotary" encoding.
            attention_window: When `attention_window > 0` the attention matrix will be zero between positions that are further apart than `attention_window`. If `attention_window == 0`, no mask is applied.
        """

        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = float(dropout)
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.attention_window = attention_window

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
            [nn.BatchNorm1d(self.embed_dim) for _ in range(n_layers)]
        )
        self.norm2 = nn.ModuleList(
            [nn.BatchNorm1d(self.embed_dim) for _ in range(n_layers)]
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
                self.embed_dim, encoding_period, 2, encoding_frequencies
            )
        elif encoding_type == "rotary":
            self.encoding = RotaryPositionalEncoding(
                self.embed_dim, encoding_period, 2, encoding_frequencies
            )
        elif encoding_type == "mlp":
            self.encoding = gnn.MLP(
                [2, 2 * self.embed_dim, self.embed_dim],
                norm="batch_norm",
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unknown positional encoding: {encoding_type}")

    @staticmethod
    def _window_mask(pos: torch.Tensor, attention_window: float) -> torch.Tensor:
        """
        Create a window mask for the transformer. The mask is a matrix of shape (B, N, N) where N is the number of
        positions and B is the batch size.

        Args:
            pos: The positions of the input tensor with shape (B, N, n_dimensions).
            attention_window: The window size.

        Returns:
            A tensor of shape (B, 1, N, N) with the window mask.
        """
        distance = torch.cdist(pos, pos)
        # divide by 2 because we want to have the window size be the full width
        mask = distance <= attention_window / 2
        return mask.unsqueeze(1)

    @staticmethod
    def _connected_mask(components: torch.Tensor) -> torch.Tensor:
        """
        Compute a mask that is zero between positions that are not connected in the graph.

        Args:
            components: (B, N) tensor with integers representing the ID of connected components. If components[i] == components[j], then i and j are connected.

        Returns:
            A tensor of shape (B, 1, N, N) with the connected mask.
        """
        mask = components.unsqueeze(1) == components.unsqueeze(2)
        return mask.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        components: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (B, N, embed_dim).
            pos: The positions of the input tensor with shape (B, N, n_dimensions).
            padding_mask: (Optional) The padding mask with shape (B, N). If None, no mask is applied.
            components: (Optional) (B, N) tensor with IDs of connected components. If i and j are connected, then components[i] == components[j]. If provided, the attention matrix will be zero between different components.
        """
        B, N = x.shape[:2]
        if isinstance(self.encoding, (AbsolutePositionalEncoding, gnn.MLP)):
            x = x + self.encoding(pos.reshape(B * N, -1)).reshape(B, N, -1)

        # We need an attention mask if there is padding, windowing or we are masking by connected components.
        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        if components is not None:
            connected_mask = self._connected_mask(components)
            attn_mask = (
                connected_mask if attn_mask is None else attn_mask & connected_mask
            )
        if self.attention_window > 0:
            window_mask = self._window_mask(pos, self.attention_window)
            attn_mask = window_mask if attn_mask is None else attn_mask & window_mask

        for i in range(self.n_layers):
            # using pre-norm transformer, so we apply layer norm before attention
            x_norm = self.norm1[i](x.reshape(B * N, -1)).reshape(B, N, -1)
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
                F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout, attn_mask=attn_mask
                )
                .transpose(1, 2)
                .reshape(B, N, self.embed_dim)
            )
            x_attention = self.Wo[i](x_attention)
            x = x + x_attention
            # compute the feed forward layer and apply residual connection, as before
            x_fc = self.norm1[i](x_attention.reshape(B * N, -1)).reshape(B, N, -1)
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
        encoding_period: float = 10.0,
        encoding_frequencies: str = "linear",
        attention_window: float = 0.0,
        connected_mask: bool = False,
    ):

        super().__init__()
        self.state_ndim = 14
        self.action_ndim = 2
        self.embed_dim = n_channels * n_heads
        self.dropout = float(dropout)
        self.connected_mask = connected_mask

        self.readin = gnn.MLP(
            [self.state_ndim, 2 * self.embed_dim, self.embed_dim],
            norm="batch_norm",
            dropout=self.dropout,
        )
        self.readout = gnn.MLP(
            [self.embed_dim, 2 * self.embed_dim, self.action_ndim],
            plain_last=True,
            norm="batch_norm",
            dropout=self.dropout,
        )
        self.transformer = Transformer(
            n_layers,
            self.embed_dim,
            n_heads,
            self.dropout,
            encoding_type=encoding_type,
            encoding_period=encoding_period,
            encoding_frequencies=encoding_frequencies,
            attention_window=attention_window,
        )

    def forward(self, data: Data) -> torch.Tensor:
        batch_size = data.batch_size if isinstance(data, Batch) else 1
        x = self.readin(data.state).reshape(batch_size, -1, self.embed_dim)
        pos = data.positions.reshape(batch_size, -1, 2)
        padding_mask = (
            data.padding_mask.reshape(batch_size, -1)
            if hasattr(data, "padding_mask")
            else None
        )
        components = (
            data.components.reshape(batch_size, -1) if self.connected_mask else None
        )
        y = self.transformer(x, pos, padding_mask, components).reshape(
            -1, self.embed_dim
        )
        return self.readout(y)


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
        self.dropout = float(dropout)

        self.readin_agent = gnn.MLP(
            [4, 2 * self.embed_dim, self.embed_dim],
            norm="batch_norm",
            dropout=self.dropout,
        )
        self.readin_target = gnn.MLP(
            [2, 2 * self.embed_dim, self.embed_dim],
            norm="batch_norm",
            dropout=self.dropout,
        )
        self.readout = gnn.MLP(
            [self.embed_dim, 2 * self.embed_dim, 1],
            plain_last=True,
            norm="batch_norm",
            dropout=self.dropout,
        )
        self.transformer = Transformer(
            n_layers,
            self.embed_dim,
            n_heads,
            self.dropout,
            encoding_type=None,
            attention_window=0,
        )

    def forward(self, action: torch.Tensor, data: Data) -> torch.Tensor:
        batch_size = data.batch_size if isinstance(data, Batch) else 1

        x_agent = torch.cat([data.positions, action], dim=-1)
        x_agent = self.readin_agent(x_agent)
        x_agent = x_agent.reshape(batch_size, -1, self.embed_dim)
        n_agents = x_agent.size(1)

        x_targets = self.readin_target(data.targets)
        x_targets = x_targets.reshape(batch_size, -1, self.embed_dim)

        # concatenate the agent and target embeddings so now we have shape (B, n_agents + n_targets, embed_dim)
        x = torch.cat([x_agent, x_targets], dim=1)
        y = self.transformer(x, x)
        # get the outputs corresponding to the agents
        y_agent = y[:, :n_agents]
        z = self.readout(y_agent.reshape(batch_size * n_agents, self.embed_dim))
        return z.squeeze(1)


class TransformerActorCritic(ActorCritic):
    def __init__(
        self,
        n_layers: int = 2,
        n_channels: int = 32,
        n_heads: int = 2,
        dropout: float = 0.0,
        encoding_type: str = "mlp",
        encoding_period: float = 10.0,
        encoding_frequencies: str = "linear",
        attention_window: float = 0.0,
        connected_mask: bool = False,
        **kwargs,
    ):
        """
        Args:
            n_layers: The number of transformer layers.
            n_channels: The number of channels in the transformer. Must be divisible by n_heads.
            n_heads: The number of attention heads. Each head will have dimension n_channels // n_heads.
            dropout: The dropout rate.
            encoding_type: The type of positional encoding to use. Either "absolute", "rotary", "mlp" or None.
            encoding_period: The period of the encoding. Only used for "absolute" and "rotary" encoding.
            encoding_frequencies: The method to generate the frequencies. Either "linear" or "geometric". Only used for "absolute" and "rotary" encoding.
            attention_window: When `attention_window > 0` the attention matrix will be zero between positions that are further apart than `attention_window`. If `attention_window == 0`, no mask is applied.
            connected_mask: If True, the attention matrix will be zero between positions that are not connected in the graph.
        """
        actor = TransformerActor(
            n_layers,
            n_channels,
            n_heads,
            dropout,
            encoding_type=encoding_type,
            encoding_period=encoding_period,
            encoding_frequencies=encoding_frequencies,
            attention_window=attention_window,
            connected_mask=connected_mask,
        )
        critic = TransformerCritic(
            n_layers,
            n_channels,
            n_heads,
            dropout,
        )
        super().__init__(actor, critic, **kwargs)
