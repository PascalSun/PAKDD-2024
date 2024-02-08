import torch
import torch.nn.functional as F
from torch_geometric.nn import CGConv


class CGC(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        # hidden_channels: List[int],
        # out_channels: int
    ):
        super(CGC, self).__init__()
        self.conv1 = CGConv(in_channels)
        self.conv2 = CGConv(
            in_channels,
        )

    def forward(self, features, edge_index):
        x = features
        x = F.relu(self.conv1(x, edge_index))  #
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x, F.log_softmax(x, dim=-1)
