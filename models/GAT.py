from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
    ):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels[0])
        self.conv2 = GATConv(hidden_channels[0], out_channels)

    def forward(self, features, edge_index):
        x = features
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        return x, F.log_softmax(x, dim=1)
