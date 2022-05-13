from reconstrain.models.base import GNN
import torch


def test_gnn():
    K = 8
    F_in = 2
    F = 16
    F_out = 4

    x = torch.rand(16, 10, F_in)
    S = torch.rand(10, 10)
    gnn = GNN(F_in=F_in, F_out=F_out, K=K, F=F, n_layers=4)
    y = gnn(x, S)

    assert y.shape == (16, 10, 4), "GNN output shape is incorrect"
