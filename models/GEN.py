# import torch
# from torch_geometric.nn import GENConv
# import torch.nn.functional as F
# from typing import List
# import torch.nn as nn
#
#
# class GEN(torch.nn.Module):
#     def __init__(self, in_channels: int, hidden_channels: List[int], out_channels: int):
#         super(GEN, self).__init__()
#         self.node_encoder = nn.Linear(in_channels, hidden_channels[0])
#         self.edge_encoder = nn.Linear(edge_attr_all.size(-1), dim)
#         self.conv1 = GENConv(dim, dim)
#         self.conv2 = GENConv(dim, dim)
#         self.fc1 = nn.Linear(dim, dataset.num_classes)
#
#     def forward(self):
#         x, edge_index, edge_attr = (
#             self.node_encoder(data.x),
#             data.edge_index,
#             self.edge_encoder(edge_attr_all),
#         )
#         x = F.relu(self.conv1(x, edge_index, edge_attr))
#         x = F.dropout(x, p=p, training=self.training)
#         x = F.relu(self.conv2(x, edge_index, edge_attr))
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=1)
