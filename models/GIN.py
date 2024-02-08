# import torch
# import torch_geometric.nn as pyg_nn
#
#
# class GIN(torch.nn.Module):
#     def __init__(self, dim=d):
#         super(GIN, self).__init__()
#         nn1 = nn.Sequential(
#             nn.Linear(dataset.num_features, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim)
#         )
#         nn2 = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
#         self.conv1 = pyg_nn.GINConv(nn1)
#         self.conv2 = pyg_nn.GINConv(nn2)
#
#         self.fc1 = nn.Linear(dim, dataset.num_classes)
#
#     def forward(self):
#         x, edge_index = data.x, data.edge_index
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=p, training=self.training)
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=1)
