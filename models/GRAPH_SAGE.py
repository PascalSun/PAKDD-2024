from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F  # noqa
from torch_geometric.nn import SAGEConv  # noqa

from src.iid.utils import reconstruct_loss
from src.utils.logger import get_logger

logger = get_logger()


class GraphSAGETask(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        aggr: str = "max",
    ):
        super(GraphSAGETask, self).__init__()
        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(SAGEConv(in_channels, hidden_channels[0], aggr=aggr))
        for i in range(1, len(hidden_channels)):
            self.conv_layers.append(
                SAGEConv(hidden_channels[i - 1], hidden_channels[i], aggr=aggr)
            )
        self.conv_layers.append(SAGEConv(hidden_channels[-1], out_channels, aggr=aggr))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def forward(self, features, edge_index):
        x = features
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i != len(self.conv_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.2)
        return x, torch.log_softmax(x, dim=-1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer

        for epoch in range(epochs + 1):
            # TODO: when to use batch to train?
            self.train()
            optimizer.zero_grad()
            _, out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            self.eval()
            _, pred = self(data.x, data.edge_index)
            pred = pred.argmax(dim=1)
            correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()  # noqa
            acc = int(correct) / int(data.val_mask.sum())

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}"
                )


class GraphSAGEEmb(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        aggr: str = "max",
    ):
        super(GraphSAGEEmb, self).__init__()
        if len(hidden_channels) == 0:
            raise ValueError("hidden_channels should have at least one element")

        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(SAGEConv(in_channels, hidden_channels[0], aggr=aggr))
        # Define hidden layers
        for idx in range(1, len(hidden_channels)):
            self.conv_layers.append(
                SAGEConv(hidden_channels[idx - 1], hidden_channels[idx], aggrr=aggr)
            )

        self.conv_layers.append(SAGEConv(hidden_channels[-1], out_channels, aggr=aggr))

    def forward(self, features, edge_index):
        """
        # Discuss the elu and dropout here
        :param features:
        :param edge_index:
        :return:
        """
        x = features
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i != len(self.conv_layers) - 1:  # if not the last layer
                x = F.elu(x)
                # TODO: dropout rate is 0.2, we can tune it later
                x = F.dropout(x, p=0.2)
        return x

    def fit(
        self, data, epochs, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> pd.DataFrame:
        if not optimizer:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(epochs):
            # TODO: it is already over fitting here when doing the reconstruction loss
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            # DISCUSS: GraphSAGE is trying to aggregate the information from the neighbors
            # so the reconstruction loss is not suitable here probably?
            loss = reconstruct_loss(out, data.edge_index)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                logger.info(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}")

        self.eval()
        with torch.no_grad():
            out = self(data.x, data.edge_index)
        emb_df = pd.DataFrame(out.detach().cpu().numpy())
        emb_df["y"] = data.y.detach().cpu().numpy()
        return emb_df
