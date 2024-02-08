from typing import List

import pandas as pd
import torch

from src.iid.utils.loss import analytical_geometry_loss
from src.utils.logger import get_logger

logger = get_logger()


class GraphAGEmb(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int], out_channels: int):
        super(GraphAGEmb, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        # add the first layer with in_channels and hidden_channels[0]
        self.conv_layers.append(torch.nn.Linear(in_channels, hidden_channels[0]))
        # add the rest of the layers
        for i in range(1, len(hidden_channels)):
            self.conv_layers.append(
                torch.nn.Linear(hidden_channels[i - 1], hidden_channels[i])
            )
        # add the last layer with hidden_channels[-1] and out_channels
        self.conv_layers.append(torch.nn.Linear(hidden_channels[-1], out_channels))

        self.optimizer = torch.optim.ASGD(
            self.parameters(), lr=0.0000001, weight_decay=5e-4
        )

    def forward(self, features, edge_index):
        x = features
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i != len(self.conv_layers) - 1:
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        return x

    def fit(self, data, epochs: int):
        # loss function will try to minimize the distance between the embeddings of the nodes
        optimizer = self.optimizer

        for epoch in range(1, epochs + 1):
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = analytical_geometry_loss(out, data.edge_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}")

        self.eval()
        with torch.no_grad():
            out = self(data.x, data.edge_index)
        emb_df = pd.DataFrame(out.detach().cpu().numpy())
        emb_df["y"] = data.y.detach().cpu().numpy()
        return emb_df
