import argparse
import typing
from typing import Callable, Type

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from motion_planning.architecture.base import ActorCritic

activation_choices: typing.Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}

SequentialLayers = list[tuple[Callable, str] | Callable]


class DeepSet(nn.Module):
    def __init__(
        self,
        in_channels_phi: int,
        in_channels_rho: int,
        out_channels: int,
        n_hidden_channels: int,
        n_layers: int,
        dropout: float,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = activation_choices[activation]()

        if n_layers < 1:
            raise ValueError("n_layers for a Deep Set must be >= 1.")

        dropout = float(dropout)

        self.phi = gnn.MLP(
            in_channels=in_channels_phi,
            out_channels=in_channels_rho,
            hidden_channels=n_hidden_channels,
            num_layers=n_layers,
            dropout=dropout,
            act=activation,
        )
        self.rho = gnn.MLP(
            in_channels=in_channels_rho,
            out_channels=out_channels,
            hidden_channels=n_hidden_channels,
            num_layers=n_layers,
            dropout=dropout,
            act=activation,
        )

    def forward(self, x):
        X_shape = x.shape + (self.rho.in_channels,)
        X = torch.zeros(X_shape)
        num_elements = int(x.size()[2] / self.phi.in_channels)
        for i in range(num_elements):
            X += self.phi(
                x[:, :, i * self.phi.in_channels : (i + 1) * self.phi.in_channels]
            )
        return self.rho(X)


class GraphFilter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_taps: int,
        aggr: str | gnn.Aggregation = "sum",
    ):
        """
        Computes a polynomial MIMO graph filters. The input and output signals can have a different number of channels.
        .. math::
            \\mathbf{y} = \\sum_{k=0}^{K} W_k S^k \\mathbf{x}

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_taps: Number of taps in the filter.
            aggr: Aggregation scheme to use. For example, "add", "sum" "mean", "min", "max" or "mul".
                In addition, can be any Aggregation module (or any string that automatically resolves to it).
        """
        super().__init__()
        self.shift = gnn.SimpleConv(
            aggr=aggr,
            combine_root=None,
        )
        self.taps = nn.ModuleList(
            [gnn.Linear(in_channels, out_channels) for _ in range(n_taps + 1)]
        )

    def forward(
        self,
        x: torch.Tensor | OptPairTensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> torch.Tensor:
        # if isinstance(x, torch.Tensor):
        #     x = (x, x)

        y = self.taps[0](x)
        for i in range(1, len(self.taps)):
            x = self.shift(x, edge_index, edge_attr, size)
            y += self.taps[i](x)
        return y


class ResidualBlock(nn.Module):
    def __init__(
        self,
        conv: Callable,
        act: Callable[[torch.Tensor], torch.Tensor] | None = None,
        norm: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Residual block with a RES+ connection.
            Norm -> Activation -> Dropout -> Conv -> Residual

        Args:
            conv: Convolutional layer with input arguments (x, type_vec, adj_t).
            dropout: Dropout probability with input arguments (x).
            act: Activation function with input arguments (x).
            norm: Normalization function with input arguments (x, type_vec).
        """
        super().__init__(**kwargs)
        self.conv = conv
        self.norm = norm
        self.act = act
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        The first argument should be the input tensor, the remaining arguments are passed through to the `conv` module.
        """

        y = x
        if self.norm is not None:
            y = self.norm(y)
        if self.act is not None:
            y = self.act(y)
        y = self.dropout(y)
        y = self.conv(y, *args, **kwargs)
        return x + y


class GCN(nn.Module):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(GCN.__name__)
        group.add_argument(
            "--n_channels",
            type=int,
            default=32,
            help="Number of hidden features on each layer. Set to -1 for lazy initialization.",
        )
        group.add_argument(
            "--n_layers", type=int, default=2, help="Number of GNN layers."
        )
        group.add_argument(
            "--activation",
            type=str,
            default="leaky_relu",
            choices=list(activation_choices),
        )
        group.add_argument(
            "--mlp_read_layers",
            type=int,
            default=1,
            help="Number of MLP layers to use for readin/readout.",
        )
        group.add_argument(
            "--mlp_per_gnn_layers",
            type=int,
            default=0,
            help="Number of MLP layers to use per GNN layer.",
        )
        group.add_argument(
            "--mlp_hidden_channels",
            type=int,
            default=256,
            help="Number of hidden features to use in the MLP layers.",
        )
        group.add_argument(
            "--dropout", type=float, default=0.0, help="Dropout probability."
        )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_taps: int,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        A simple GNN model with a readin and readout MLP. The structure of the architecture is expressed using hyperparameters. This allows for easy hyperparameter search.

        Args:
            in_channels: Number of input features. Set to -1 for lazy initialization.
            out_channels: Number of output features.
            n_taps: Number of filter taps per layer.
            n_layers: Number of GNN layers.
            n_channels: Number of hidden features on each layer.
            n_taps: Number of filter taps per layer.
            activation: Activation function to use.
            read_layers: Number of MLP layers to use for readin/readout.
            read_hidden_channels: Number of hidden features to use in the MLP layers.
            residual: Type of residual connection to use: "res", "res+", "dense", "plain".
                https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGCNLayer.html
            normalization: Type of normalization to use: "batch" or "layer".
        """
        super().__init__()
        if isinstance(activation, str):
            activation = activation_choices[activation]()

        if mlp_read_layers < 1:
            raise ValueError("mlp_read_layers must be >= 1.")

        # ensure that dropout is a float
        dropout = float(dropout)

        # Readin MLP: Changes the number of features from in_channels to n_channels
        self.readin = gnn.MLP(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=n_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            norm=None,
            plain_last=False,
        )

        # Readout MLP: Changes the number of features from n_channels to out_channels
        self.readout = gnn.MLP(
            in_channels=n_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            norm=None,
            plain_last=True,
        )

        # GNN layers operate on n_channels features
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_layers):
            conv: SequentialLayers = [
                (
                    GraphFilter(
                        in_channels=n_channels,
                        out_channels=n_channels,
                        n_taps=n_taps,
                    ),
                    "x, edge_index, edge_attr, size -> x",
                ),
            ]
            if mlp_per_gnn_layers > 0:
                conv += [
                    (activation, "x -> x"),
                    (
                        gnn.MLP(
                            in_channels=n_channels,
                            hidden_channels=mlp_hidden_channels,
                            out_channels=n_channels,
                            num_layers=mlp_per_gnn_layers,
                            dropout=dropout,
                            act=activation,
                            norm=None,
                            plain_last=True,
                        ),
                        "x -> x",
                    ),
                ]
            self.residual_blocks += [
                (
                    ResidualBlock(
                        conv=gnn.Sequential("x, edge_index, edge_attr, size", conv),
                        # norm=gnn.BatchNorm(n_channels),
                        norm=None,
                        act=activation,
                        dropout=dropout,
                    )
                )
            ]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> torch.Tensor:
        x = self.readin(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x, edge_index, edge_attr, size)
        x = self.readout(x)
        return x


class GNNActorCritic(ActorCritic):
    def __init__(
        self,
        state_ndim: int = 14,
        action_ndim: int = 2,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        **kwargs,
    ):
        actor = GCN(
            state_ndim,
            action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )
        critic = GCN(
            state_ndim + action_ndim,
            1,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )
        super().__init__(actor, critic)

    def forward_actor(self, data: Data) -> torch.Tensor:
        """
        Returns normalized action within the range [-1, 1].
        """
        return self.actor(data.state, data.edge_index).tanh()

    def forward_critic(self, action: torch.Tensor, data: Data) -> torch.Tensor:
        x = torch.cat([data.state, action], dim=-1)
        return self.critic(x, data.edge_index)
