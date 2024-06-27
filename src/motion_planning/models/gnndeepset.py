import argparse
import typing
from typing import Callable, Type

import numpy as np
import scipy

import pytorch_lightning as pl
import scipy.spatial
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.conv import GATv2Conv
from torchcps.gnn import GraphFilter, ResidualBlock
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

activation_choices: typing.Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
}

SequentialLayers = list[tuple[Callable, str] | Callable]


class GraphAttentionFilter(gnn.MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        radius: float,
        n_layers: int,
        dropout: float,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
    ):
        super().__init__(aggr="sum")
        if isinstance(activation, str):
            activation = activation_choices[activation]()

        if n_layers < 1:
            raise ValueError("n_layers for a Deep Set must be >= 1.")

        dropout = float(dropout)

        self.radius = radius

        self.conv = GATv2Conv(
            in_channels,
            out_channels,
            heads,
            dropout=dropout
        )
        
    def forward(self, positions: torch.Tensor):
        edge_index = gnn.radius_graph(
            positions,
            self.radius,
            max_num_neighbors=10,
            flow='target_to_source'
        )
        x = self.conv(positions, edge_index)
        return x
    

class GraphAttentionFilterTargets(GraphAttentionFilter):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        radius: float,
        n_layers: int,
        dropout: float,
        activation: typing.Union[nn.Module, str] = "leaky_relu",
    ):
        super().__init__(in_channels, out_channels, heads, radius, n_layers, dropout, activation)
    
    def forward(self, agent_positions: torch.Tensor, target_positions: torch.Tensor):
        n_agents = agent_positions.shape[0]
        n_targets = target_positions.shape[0]
        
        target_edges = gnn.radius_graph(
            torch.cat([agent_positions, target_positions], dim=0),
            self.radius,
            max_num_neighbors=10,
            flow='target_to_source',
        )
        mask = torch.logical_and(target_edges[0] >= n_agents, target_edges[1] < n_agents)
        target_edges = target_edges[:, mask]
        x = torch.cat([agent_positions, target_positions], dim=0)
        x = self.conv(x, target_edges)
        x, _ = torch.split(x, [n_agents, n_targets], dim=0)
        return x


class GCNDeepSet(nn.Module):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(GCNDeepSet.__name__)
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
            help="Number of Deep Set/MLP layers to use for readin/readout.",
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
        radius: float,
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

        # Readin Deep Set: Changes the number of features to n_channels
        n_channels_ds = int(n_channels / 2)

        self.agent_preprocessor = GraphAttentionFilter(
            in_channels=2,
            out_channels=n_channels_ds,
            heads=1,
            radius=radius,
            n_layers=mlp_read_layers,
            dropout=dropout,
            activation=activation,
        )

        self.target_preprocessor = GraphAttentionFilterTargets(
            in_channels=2,
            out_channels=n_channels_ds,
            heads=1,
            radius=radius,
            n_layers=mlp_read_layers,
            dropout=dropout,
            activation=activation,
        )

        self.self_preprocessor = gnn.MLP(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=n_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            plain_last=True,
        )

        self.readout = gnn.MLP(
            in_channels=n_channels*2,
            hidden_channels=mlp_hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_read_layers,
            dropout=dropout,
            act=activation,
            plain_last=True,
        )

        # GNN layers operate on n_channels features
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_layers):
            conv: SequentialLayers = [
                (
                    GraphFilter(
                        in_channels=n_channels*2,
                        out_channels=n_channels*2,
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
                            in_channels=n_channels*2,
                            hidden_channels=mlp_hidden_channels,
                            out_channels=n_channels*2,
                            num_layers=mlp_per_gnn_layers,
                            dropout=dropout,
                            act=activation,
                            plain_last=True,
                        ),
                        "x -> x",
                    ),
                ]
            self.residual_blocks += [
                (
                    ResidualBlock(
                        conv=gnn.Sequential("x, edge_index, edge_attr, size", conv),
                        norm=gnn.BatchNorm(n_channels*2),
                        act=activation,
                        dropout=dropout,
                    )
                )
            ]

    def forward(self,
                state: list,
                edge_index: Adj,
                edge_attr: OptTensor = None,
                size: Size = None):
        own_obs = state[0]
        agent_obs = state[1]
        target_obs = state[2]
        s = self.self_preprocessor(own_obs)
        a = self.agent_preprocessor(agent_obs)
        t = self.target_preprocessor(agent_obs, target_obs)
        x = torch.cat([s, a, t], dim=1)
        for residual_block in self.residual_blocks:
            x = residual_block(x, edge_index, edge_attr, size)
        x = self.readout(x)
        return x
