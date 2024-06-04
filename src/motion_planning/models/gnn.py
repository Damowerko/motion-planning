import argparse
import typing
from typing import Callable, Type

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

activation_choices: typing.Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}

SequentialLayers = list[tuple[Callable, str] | Callable]


class EGraphFilter(gnn.MessagePassing):
    def __init__(
        self,
        node_features_in: int,
        node_features_out: int,
        coord_features: int,
        edge_features: int,
        hidden_features: int,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
    ):
        super().__init__(aggr='sum')
        if isinstance(activation, str):
            activation = activation_choices[activation]

        self.node_features_in = node_features_in
        self.node_features_out = node_features_out
        self.coord_features = coord_features
        self.edge_features = edge_features

        self.edge_in_norm = nn.BatchNorm1d(2 * node_features_in + 1)

        self.node_mlp = gnn.MLP(
            [node_features_in + edge_features, hidden_features, node_features_out],
            act=activation()
        )

        final_coord_layer = nn.Linear(hidden_features, coord_features, bias=False)
        nn.init.xavier_normal_(final_coord_layer.weight, gain=1e-3)
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_features, hidden_features),
            activation(),
            final_coord_layer,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_features_in + 1, hidden_features),
            activation(),
            nn.Linear(hidden_features, edge_features),
            activation(),
        )

    def forward(
        self,
        h: torch.Tensor | OptPairTensor,
        x: torch.Tensor | OptPairTensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> torch.Tensor:
        feat = self.propagate(edge_index, h=h, x=x)
        m, x_n = torch.split(feat, [self.edge_features, self.coord_features], dim=1)
        x_n = x + x_n
        h_n = self.node_mlp(torch.cat([h, m], dim=1))
        return h_n, x_n
    
    def message(self, h_i, h_j, x_i, x_j):
        m_ij = torch.cat([h_i, h_j, torch.norm(x_i - x_j, dim=1, keepdim=True)], dim=1)
        m_ij = self.edge_in_norm(m_ij)
        m_ij = self.edge_mlp(m_ij)
        x_ij = (x_i - x_j) * self.coord_mlp(m_ij)
        return torch.cat([m_ij, x_ij], dim=1)


class EGCN(nn.Module):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(EGCN.__name__)
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
        coord_channels: int,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
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

        self.in_channels = in_channels
        self.coord_channels = coord_channels

        # Readin MLP: Changes the number of features from in_channels to n_channels
        self.readin = gnn.MLP(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=n_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
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
            plain_last=True,
        )

        # GNN layers operate on n_channels features
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = [
                (
                    EGraphFilter(
                        node_features_in=n_channels,
                        node_features_out=n_channels,
                        coord_features=coord_channels,
                        edge_features=n_channels,
                        hidden_features=n_channels,
                    ),
                    "h, x, edge_index, edge_attr, size -> h, x",
                ),
            ]
            self.convs += [gnn.Sequential("h, x, edge_index, edge_attr, size", conv)]

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> torch.Tensor:
        h = self.readin(h)
        for conv in self.convs:
            h, x = conv(h, x, edge_index, edge_attr, size)
        h = self.readout(h)
        return torch.concat([h, x], dim=1)
