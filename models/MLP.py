# from torch import nn
#
#
# class MLP(nn.Module):
#     def __init__(self, hidden_dim=d):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(dataset.num_features, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, dataset.num_classes)
#
#     def forward(self):
#         x = F.relu(self.fc1(data.x))
#         x = F.dropout(x, p=p, training=self.training)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
