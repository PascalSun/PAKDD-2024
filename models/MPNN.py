# class MPNN(torch.nn.Module):
#     def __init__(self, dim=d):
#         super(MPNN, self).__init__()
#         nn1 = nn.Sequential(
#             nn.Linear(edge_attr_all.shape[1], 16),
#             nn.ReLU(),
#             nn.Linear(16, dataset.num_features * dim),
#         )
#         self.conv1 = pyg_nn.NNConv(dataset.num_features, dim, nn1)
#         nn2 = nn.Sequential(
#             nn.Linear(edge_attr_all.shape[1], 16), nn.ReLU(), nn.Linear(16, dim * dim)
#         )
#         self.conv2 = pyg_nn.NNConv(dim, dim, nn2)
#         self.fc1 = nn.Linear(dim, dataset.num_classes)
#
#     def forward(self):
#         x, edge_index, edge_attr = (
#             data.x,
#             data.edge_index,
#             edge_attr_all,
#         )  # data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_attr))
#         x = F.dropout(x, p=p, training=self.training)
#         x = F.relu(self.conv2(x, edge_index, edge_attr))
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=1)
