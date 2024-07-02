import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# class GNNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, num_layers=2):
#         super(GNNEncoder, self).__init__()
#
#         self.num_layers = num_layers
#         self.convs = torch.nn.ModuleList()
#
#         # Input layer
#         self.convs.append(GCNConv(in_channels, 128))
#         # Hidden layers
#         for _ in range(num_layers - 2):
#             self.convs.append(GCNConv(128, 128))
#         # Output layer
#         self.convs.append(GCNConv(128, out_channels))
#
#     def forward(self, x, edge_index):
#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=0.5, training=self.training)
#         return x

class GNNEncoder(torch.nn.Module):  # Update: 加了一个dropout rate parameter. 20240530
    def __init__(self, in_channels, out_channels, num_layers=2, dropout=0.5):
        super(GNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, 128))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(128, 128))
        # Output layer
        self.convs.append(GCNConv(128, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x



class MOELayer(torch.nn.Module):
    def __init__(self, in_channels, num_experts):
        super(MOELayer, self).__init__()

        self.gates = torch.nn.Linear(in_channels, num_experts)

    def forward(self, x):
        # Softmax to determine the weight for each expert
        weights = F.softmax(self.gates(x), dim=1)
        return weights


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.decode = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.decode(x)
