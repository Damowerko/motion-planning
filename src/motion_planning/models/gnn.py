import argparse
import typing
from typing import Callable, Type, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

SequentialLayers = list[tuple[Callable, str] | Callable]


class ModReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor((1,)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.b)
    
    def forward(self, x: torch.Tensor):
        return torch.nn.functional.relu((torch.abs(x) + self.b).to(dtype=torch.float32)) * x / torch.abs(x)
    

# FINISH THIS IMPLEMENTATION
class ComplexBatchNorm1d(nn.Module):
    running_mean: Optional[torch.Tensor]

    def __init__(
        self,
        num_features,
        eps=1e-3,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats


activation_choices: typing.Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "mod_relu": ModReLU,
}


class ComplexGraphFilter(nn.Module):
    def __init__(
        self,
        f_channels: int,
        g_channels: int,
        mlp_hidden_channels: int,
        activation: str | nn.Module = "mod_relu",
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

        if isinstance(activation, str):
            activation = activation_choices[activation_choices]

        self.shift = gnn.SimpleConv(
            aggr=aggr,
            combine_root=None,
        )
        self.equi_mlp = nn.Sequential(
            nn.Linear(g_channels, mlp_hidden_channels),
            activation,
            nn.Linear(mlp_hidden_channels, f_channels),
        )
        self.inv_mlp = nn.Sequential(
            nn.Linear(f_channels, mlp_hidden_channels),
            activation,
            nn.Linear(mlp_hidden_channels, g_channels),
        )

    def forward(
        self,
        f: torch.Tensor | OptPairTensor,
        g: torch.Tensor | OptPairTensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> torch.Tensor:
        # if isinstance(x, torch.Tensor):
        #     x = (x, x)

        f_prime = self.shift(f, edge_index, edge_attr, size)
        g_prime = self.shift(g, edge_index, edge_attr, size)
        f_n = f + self.equi_mlp(g_prime)
        g_n = g + self.inv_mlp(f_prime)
        return f_n, g_n


class ComplexGCN(nn.Module):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(ComplexGCN.__name__)
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
            default="mod_relu",
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
        f_channels_in: int,
        g_channels_in: int,
        f_channels_out: int,
        g_channels_out: int,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: typing.Union[nn.Module, str] = "mod_relu",
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

        # Readin MLPs: Changes the number of features from in_channels to n_channels
        # self.readin_f = gnn.MLP(
        #     in_channels=f_channels_in,
        #     hidden_channels=mlp_hidden_channels,
        #     out_channels=n_channels,
        #     num_layers=mlp_read_layers,
        #     dropout=dropout,
        #     act=activation,
        #     plain_last=False,
        # )
        self.readin_f = nn.Sequential(
            nn.Linear(f_channels_in, mlp_hidden_channels),
            activation,
            nn.Linear(mlp_hidden_channels, n_channels),
        )

        # self.readin_g = gnn.MLP(
        #     in_channels=g_channels_in,
        #     hidden_channels=mlp_hidden_channels,
        #     out_channels=n_channels,
        #     num_layers=mlp_read_layers,
        #     dropout=dropout,
        #     act=activation,
        #     plain_last=False,
        # )
        self.readin_g = nn.Sequential(
            nn.Linear(g_channels_in, mlp_hidden_channels),
            activation,
            nn.Linear(mlp_hidden_channels, n_channels),
        )

        # Readout MLP: Changes the number of features from n_channels to out_channels
        # self.readout_f = gnn.MLP(
        #     in_channels=n_channels,
        #     hidden_channels=mlp_hidden_channels,
        #     out_channels=f_channels_out,
        #     num_layers=mlp_read_layers,
        #     dropout=dropout,
        #     act=activation,
        #     plain_last=True,
        # )
        self.readout_f = nn.Sequential(
            nn.Linear(n_channels, mlp_hidden_channels),
            activation,
            nn.Linear(mlp_hidden_channels, f_channels_out),
        )

        # self.readout_g = gnn.MLP(
        #     in_channels=n_channels,
        #     hidden_channels=mlp_hidden_channels,
        #     out_channels=g_channels_out,
        #     num_layers=mlp_read_layers,
        #     dropout=dropout,
        #     act=activation,
        #     plain_last=True,
        # )
        self.readout_g = nn.Sequential(
            nn.Linear(n_channels, mlp_hidden_channels),
            activation,
            nn.Linear(mlp_hidden_channels, g_channels_out),
        )

        # GNN layers operate on n_channels features
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv: SequentialLayers = [
                (
                    ComplexGraphFilter(
                        f_channels=n_channels,
                        g_channels=n_channels,
                        mlp_hidden_channels=mlp_hidden_channels,
                        activation=activation,
                    ),
                    "f, g, edge_index, edge_attr, size -> f, g",
                ),
            ]
            self.convs += [gnn.Sequential("f, g, edge_index, edge_attr, size", conv)]

    def forward(
        self,
        f: torch.Tensor,
        g: torch.Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> torch.Tensor:
        f = self.readin_f(f)
        g = self.readin_g(g)
        for conv in self.convs:
            f, g = conv(f, g, edge_index, edge_attr, size)
        f = self.readout_f(f)
        g = self.readout_g(g)
        return f, g