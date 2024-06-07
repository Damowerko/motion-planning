import argparse
import typing
from typing import Callable, Type, Optional

import pytorch_lightning as pl
import numpy as np
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
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.complex128)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:,0] = 1.0 / np.sqrt(2)
            self.running_covar[:,1] = 1.0 / np.sqrt(2)
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.0 / np.sqrt(2)
            self.running_covar[:,1] = 1.0 / np.sqrt(2)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:,:2], 1.0 / np.sqrt(2))
            nn.init.zeros_(self.weight[:,2])
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor):
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        
        if self.training or (not self.track_running_stats):
            mean = x.mean(dim=0)
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
        
        x = x - mean[None, ...]

        if self.training or (not self.track_running_stats):
            n = x.numel() / x.size(1)
            Crr = x.real.var(dim=0, unbiased=False) + self.eps
            Cii = x.imag.var(dim=0, unbiased=False) + self.eps
            Cri = (x.real.mul(x.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:,0] + self.eps
            Cii = self.running_covar[:,1] + self.eps
            Cri = self.running_covar[:,2]
        
        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n-1) + (1 - exponential_average_factor) * self.running_covar[:,0]
                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n-1) + (1 - exponential_average_factor) * self.running_covar[:,1]
                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n-1) + (1 - exponential_average_factor) * self.running_covar[:,2]
        
        delta = Crr * Cii - Cri * Cri
        s = torch.sqrt(delta)
        t = torch.sqrt(Crr + Cii + 2 * s)
        inverse_det = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_det
        Rii = (Crr + s) * inverse_det
        Rri = -Cri * inverse_det

        x = (Rrr[None,:] * x.real + Rri[None,:] * x.imag).type(torch.complex128) + 1j * (Rri[None,:] * x.real + Rii[None,:] * x.imag).type(torch.complex128)

        if self.affine:
            x = (self.weight[None,:,0] * x.real + self.weight[None,:,2] * x.imag + self.bias[None,:,0]).type(torch.complex128) + 1j * (self.weight[None,:,2] * x.real + self.weight[None,:,1] * x.imag + self.bias[None,:,1]).type(torch.complex128)

        return x

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
        layers = [nn.Linear(f_channels_in, mlp_hidden_channels), activation]
        for _ in range(1, mlp_read_layers-1):
            layers += [nn.Linear(mlp_hidden_channels, mlp_hidden_channels), activation]
        layers += [nn.Linear(mlp_hidden_channels, n_channels)]
        self.readin_f = nn.Sequential(*layers)

        # self.readin_g = gnn.MLP(
        #     in_channels=g_channels_in,
        #     hidden_channels=mlp_hidden_channels,
        #     out_channels=n_channels,
        #     num_layers=mlp_read_layers,
        #     dropout=dropout,
        #     act=activation,
        #     plain_last=False,
        # )
        layers = [nn.Linear(g_channels_in, mlp_hidden_channels), activation]
        for _ in range(1, mlp_read_layers-1):
            layers += [nn.Linear(mlp_hidden_channels, mlp_hidden_channels), activation]
        layers += [nn.Linear(mlp_hidden_channels, n_channels)]
        self.readin_g = nn.Sequential(*layers)

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
        layers = [nn.Linear(n_channels, mlp_hidden_channels), activation]
        for _ in range(1, mlp_read_layers-1):
            layers += [nn.Linear(mlp_hidden_channels, mlp_hidden_channels), activation]
        layers += [nn.Linear(mlp_hidden_channels, f_channels_out)]
        self.readout_f = nn.Sequential(*layers)

        # self.readout_g = gnn.MLP(
        #     in_channels=n_channels,
        #     hidden_channels=mlp_hidden_channels,
        #     out_channels=g_channels_out,
        #     num_layers=mlp_read_layers,
        #     dropout=dropout,
        #     act=activation,
        #     plain_last=True,
        # )
        layers = [nn.Linear(n_channels, mlp_hidden_channels), activation]
        for _ in range(1, mlp_read_layers-1):
            layers += [nn.Linear(mlp_hidden_channels, mlp_hidden_channels), activation]
        layers += [nn.Linear(mlp_hidden_channels, g_channels_out)]
        self.readout_g = nn.Sequential(*layers)

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