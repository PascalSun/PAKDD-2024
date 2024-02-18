from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# from iid.utils import reconstruct_loss

# analytical_geometry_loss
from utils.logger import get_logger

logger = get_logger()


class GCNTask(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int], out_channels: int):
        super(GCNTask, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(in_channels, hidden_channels[0]))
        for i in range(1, len(hidden_channels)):
            self.conv_layers.append(GCNConv(hidden_channels[i - 1], hidden_channels[i]))
        self.conv_layers.append(GCNConv(hidden_channels[-1], out_channels))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1, weight_decay=5e-4)

    def forward(self, features, edge_index):
        x = features
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i != len(self.conv_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2)
        return x, torch.log_softmax(x, dim=-1)

    def fit(self, data, epochs: int):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer
        for epoch in range(1, epochs + 1):
            self.train()
            optimizer.zero_grad()
            _, out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            self.eval()
            _, pred = self(data.x, data.edge_index)
            pred = pred.argmax(dim=1)
            correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
            acc = int(correct) / int(data.val_mask.sum())

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}"
                )


class GCNEmb(torch.nn.Module):
    def __init__(
            self, in_channels: int, hidden_layer_dim: List[int], out_channels: int
    ):
        super(GCNEmb, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(in_channels, hidden_layer_dim[0]))
        for i in range(1, len(hidden_layer_dim)):
            self.conv_layers.append(
                GCNConv(hidden_layer_dim[i - 1], hidden_layer_dim[i])
            )
        self.conv_layers.append(GCNConv(hidden_layer_dim[-1], out_channels))
        logger.info(f"GCNEmb: {self.conv_layers}")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)

    def forward(self, features, edge_index):
        x = features
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i != len(self.conv_layers) - 1:
                x = F.relu(x)
                # DISCUSS: dropout rate will affect the performance quite a lot
                x = F.dropout(x, p=0.2)
        return x

    def fit(self, data, epochs: int):
        for epoch in range(1, epochs + 1):
            self.train()
            self.optimizer.zero_grad()
            z = self(data.x, data.edge_index)

            loss = reconstruct_loss(z, data.edge_index)
            # loss = analytical_geometry_loss(z, data.edge_index)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                logger.info(f"Epoch: {epoch:03d}, Loss: {loss:.3f}")
                break

        self.eval()
        with torch.no_grad():
            z = self(data.x, data.edge_index)

        emb_df = pd.DataFrame(z.detach().cpu().numpy())
        emb_df["y"] = data.y.detach().cpu().numpy()
        return emb_df
