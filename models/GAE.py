import pandas as pd
import torch
from torch_geometric.nn import GAE  # noqa Graph Autoencoder

from src.utils.logger import get_logger

logger = get_logger()


def train_gae(
    data,
    encoder,
    device: torch.device = torch.device("cpu"),
    epochs: int = 100,
):
    """
    :param data:
    :param encoder:
    :param device:
    :param epochs:
    :return:
    """

    features, train_pos_edge_index = data.x.to(device), data.edge_index.to(device)

    # Model and Optimizer Setup
    model = GAE(
        encoder=encoder,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(features, train_pos_edge_index)

        """
        the loss function here is trying to do the following:
        1. reconstruct the graph using the embedding with edge_index, then should have a probability of 1 sum
        2. reconstruct the graph using the embedding with train_neg_edge_index, then should have a probability of 0 sum

        what we can adjust here is how we construct the train_neg_edge_index
        # It depends on the graph
        # TODO: switch the negative sampling method, try to test the high hops away negative sampling
        """
        # train_neg_edge_index = negative_sampling_hop_away(
        #     nx_graph, gcn_negative_samples, neg_hops=gcn_hops
        # )
        gcn_loss = model.recon_loss(z, train_pos_edge_index)
        # gcn_loss = model.recon_loss(
        #     z, train_pos_edge_index, neg_edge_index=train_neg_edge_index
        # )
        if epoch % 10 == 0:
            logger.info(f"loss: {gcn_loss}")
        gcn_loss.backward()
        optimizer.step()

    # Assuming your model has been trained already
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to calculate gradients
        # gcn_embeddings = model.encode(data.x, train_pos_edge_index)
        embeddings = model.encode(data.x, train_pos_edge_index)

    embeddings_df = pd.DataFrame(embeddings.detach().cpu().numpy())
    # match the embedding to the original nodes
    embeddings_df = embeddings_df.copy(deep=True)
    embeddings_df["y"] = data.y.detach().cpu().numpy()
    return embeddings_df
