import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.linear = nn.Linear(in_channels * kernel_size, out_channels, bias=bias)

    def forward(self, x, S):
        """
        Args:
            x: [batch_size, num_nodes, in_channels]
            S: [num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.shape
        Sx = torch.zeros(self.kernel_size, batch_size, num_nodes, self.in_channels)
        for i in range(self.kernel_size):
            Sx[i] = S @ x
        Sx = Sx.view(batch_size, num_nodes, self.in_channels * self.kernel_size)
        return self.linear(Sx)


class GNN(nn.Module):
    def __init__(self, K: list[int], F: list[int], nonlinearity=nn.ReLU()):
        super().__init__()
        self.K = K
        self.F = F
        self.nonlinearity = nonlinearity

        assert len(self.K) == len(self.F) - 1
        self.filters = nn.ModuleList(
            [GraphConv(self.F[i], self.F[i + 1], self.K[i]) for i in range(len(self.K))]
        )

    def forward(self, x, S):
        for i in range(len(self.filters)):
            x = self.filters[i](x, S)
            # don't apply nonlinearity to the last layer
            if i < len(self.filters) - 1:
                x = self.nonlinearity(x)
        return x
