# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_geometric.nn as pyg_nn
#
# class TRAVELNet(torch.nn.Module):
#     def __init__(self, dim=d):
#         super(TRAVELNet, self).__init__()
#         convdim = 8
#         self.node_encoder = nn.Sequential(
#             nn.Linear(data.x.size(-1), dim), nn.LeakyReLU(), nn.Linear(dim, dim)
#         )
#         self.edge_encoder_dir = nn.Sequential(
#             nn.Linear(data.component_dir.size(-1), dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#         )
#         self.edge_encoder_ang = nn.Sequential(
#             nn.Linear(data.component_ang.size(-1), dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#         )
#         nn1 = nn.Sequential(
#             nn.Linear(dim + dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, convdim),
#         )
#         self.conv1 = TRAVELConv(dim, convdim, nn1)
#         nn2 = nn.Sequential(
#             nn.Linear(2 * convdim + dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dataset.num_classes),
#         )
#         self.conv2 = TRAVELConv(2 * convdim, dataset.num_classes, nn2)
#         self.bn1 = nn.BatchNorm1d(convdim * 2)
#         nn1_2 = nn.Sequential(
#             nn.Linear(dim + dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, convdim),
#         )
#         self.conv1_2 = TRAVELConv(dim, convdim, nn1_2)
#         nn2_2 = nn.Sequential(
#             nn.Linear(2 * convdim + dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dataset.num_classes),
#         )
#         self.conv2_2 = TRAVELConv(2 * convdim, dataset.num_classes, nn2_2)
#         self.bn2 = nn.BatchNorm1d(dataset.num_classes * 2)
#         self.fc = nn.Linear(dataset.num_classes * 2, dataset.num_classes)
#
#     def forward(self):
#         x, edge_index = self.node_encoder(data.x), data.edge_index
#         edge_attr_dir, edge_attr_ang = self.edge_encoder_dir(
#             data.component_dir
#         ), self.edge_encoder_ang(data.component_ang)
#         x1 = F.relu(self.conv1(x, edge_index, edge_attr_dir))
#         x2 = F.relu(self.conv1_2(x, edge_index, edge_attr_ang))
#         x = torch.cat((x1, x2), axis=1)
#         x = self.bn1(x)
#         x = F.dropout(x, p=p, training=self.training)
#         x1 = F.relu(self.conv2(x, edge_index, edge_attr_dir))
#         x2 = F.relu(self.conv2_2(x, edge_index, edge_attr_ang))
#         x = torch.cat((x1, x2), axis=1)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)
