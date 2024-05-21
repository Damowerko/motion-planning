import argparse
import typing
from typing import Callable, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data.data import BaseData
from torchcps.gnn import GCN
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

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

        self.phi = gnn.MLP(in_channels=in_channels_phi,
                       out_channels=in_channels_rho,
                       hidden_channels=n_hidden_channels,
                       num_layers=n_layers,
                       dropout=dropout,
                       act=activation,)
        self.rho = gnn.MLP(in_channels=in_channels_rho,
                       out_channels=out_channels,
                       hidden_channels=n_hidden_channels,
                       num_layers = n_layers,
                       dropout=dropout,
                       act=activation,)
        
        self.to('cuda:0')
        
    def to(self, device):
        self.device = device
        self.phi.to(device)
        self.rho.to(device)
        return super().to(device)
        
    def forward(self, x):
        X = torch.zeros((x.shape[0], self.rho.in_channels), device=self.device)
        num_elements = int(x.shape[1] / self.phi.in_channels)
        for i in range(num_elements):
            X += self.phi(x[:,i*self.phi.in_channels:(i+1)*self.phi.in_channels])
        return self.rho(X)


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

        # number of states that are not agents
        self.state_ndim = 4
        self.targets_obs_ndim = 6

        # Readin Deep Set: Changes the number of features from in_channels to n_channels
        self.agent_obs_rho = DeepSet(
            in_channels_phi=2,
            in_channels_rho=mlp_hidden_channels,
            out_channels=2,
            n_hidden_channels=mlp_hidden_channels,
            n_layers=mlp_read_layers,
            dropout=dropout,
            activation=activation,
        )

        self.gnn = GCN(
            in_channels,
            out_channels,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    def forward(self, state: torch.Tensor, edge_index, edge_attr):
        a = self.agent_obs_rho(state[:,self.state_ndim+self.targets_obs_ndim:])
        state = torch.cat((state[:,:self.state_ndim+self.targets_obs_ndim], a), axis=-1)
        action = self.gnn.forward(state, edge_index, edge_attr)
        return action
