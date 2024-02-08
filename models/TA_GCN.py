# from typing import List
#
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import TAGConv
#
# from src.utils.logger import get_logger
#
# logger = get_logger()
#
#
# class TAGCN(torch.nn.Module):
#     def __init__(self, hidden_dim=d):
#         super(TAGCN, self).__init__()
#         self.conv1 = TAGConv(dataset.num_features, hidden_dim)
#         self.conv2 = TAGConv(hidden_dim, hidden_dim)
#         self.fc1 = nn.Linear(hidden_dim, dataset.num_classes)
#
#     def forward(self):
#         x, edge_index = data.x, data.edge_index
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=p, training=self.training)
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=1)
