from reconstrain.gnn import GraphConv, GNN
import torch


def test_filter():
    x = torch.rand(16, 10, 2)
    S = torch.rand(10, 10)
    filter = GraphConv(2, 5, 4)
    y = filter(x, S)
    assert y.shape == (16, 10, 5)


def test_gnn():
    x = torch.rand(16, 10, 2)
    S = torch.rand(10, 10)
    K = [4, 8]
    F = [2, 8, 4]
    gnn = GNN(K, F, torch.relu)
    y = gnn(x, S)
    assert y.shape == (16, 10, 4)
