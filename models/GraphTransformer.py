import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(
            in_channels,
            hidden_channels,
        )
        self.conv2 = TransformerConv(hidden_channels, out_channels)

    def forward(self, features, edge_index):
        x, edge_index = features, edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        return F.log_softmax(x, dim=1)
