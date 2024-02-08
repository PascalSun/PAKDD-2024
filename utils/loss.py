import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from utils.logger import get_logger

logger = get_logger()
EPS = 1e-15


def reconstruct_loss(
        emb, edge_index, nx_graph=None, negative_samples_no=None, neg_hops: int = 2
):
    """
    we first compute the loss for positive edges, and then sample a number of negative edges and compute the loss for them as well.
    the way we calculate the pos_loss is by taking the element-wise mul of the embeddings of the source and target nodes and then passing it through a sigmoid function.
    so shape for tensor transformation is:
    (num_nodes, num_dim) * (num_nodes, num_dim) => element-wise mul (num_nodes, num_dim) => sum: (num_nodes, 1)
    => sigmoid: (num_nodes [0,1], 1) => -log: (num_nodes [-inf, 0], 1) =>  mean: (1 [-inf, 0], 1)
    objective is to maximize the loss to 0

    to calculate the neg_loss, we sample a number of negative edges and compute the loss for them as well.
    so the shape for tensor transformation is:
    (num_negative_samples, num_dim) * (num_negative_samples, num_dim) => element-wise mul (num_negative_samples, num_dim) => sum: (num_negative_samples, 1) => 1 - sigmoid: (num_negative_samples [0,1], 1) => -log: (num_negative_samples [-inf, 0], 1) =>  mean: (1 [-inf, 0], 1)

    :param nx_graph:
    :param emb:
    :param edge_index:
    :param negative_samples_no:
    :param neg_hops:
    :return:
    """
    # DISCUSS:
    # IMPORTANT: This loss function will get to 0
    source, target = edge_index
    logits = (emb[source] * emb[target]).sum(-1)
    pos_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    # pos_loss = -torch.log(torch.sigmoid((emb[source] * emb[target]).sum(-1))).mean()
    # logger.info(f"pos_loss: {pos_loss}")

    """
    Negative sampling

    1. totally random negative sampling
    num_nodes = embeddings.shape[0]

    neg_source = torch.randint(
        0, num_nodes, (num_negative_samples,), device=embeddings.device
    )
    neg_target = torch.randint(
        0, num_nodes, (num_negative_samples,), device=embeddings.device
    )
    2. negative sampling hops away from the node

    Compared to the first one, the second on perform a bit better.

    There is not much difference between 1-hop and 2-hop. But 3-hop is a bit worse than 1-hop and 2-hop.
    """

    # neg_source, neg_target = negative_sampling_hop_away(
    #     nx_graph, negative_samples_no, neg_hops
    # )
    # logger.debug(f"neg_source: {neg_source}")
    # logger.debug(f"neg_target: {neg_target}")
    #
    # neg_loss = -torch.log(
    #     1 - torch.sigmoid((emb[neg_source] * emb[neg_target]).sum(-1))
    # ).mean()

    return pos_loss


def analytical_geometry_loss(emb, edge_index):
    """
    We want to make sure the one hop neighbors are close to each other, then two hops

    So we will calculate the distance between the embeddings of the one hop neighbors

    dist(emb[n_hop_source], emb[n_hop_target]) as D
    N is the number of hops

    the loss function will be:
            log(floor(log_e(D)) / N) >= 0

    :param emb:
    :param edge_index:
    :return:
    """
    # edge_index is one hop neighbors

    # Update: 2023.09.19, get this one to be probability based loss, so the probability of two nodes
    # has link is decay exponentially with the distance between them, in which we can have the training objective
    """
    We want to make sure the one hop neighbors are close to each other, then two hops

    :param emb:
    :param edge_index:
    :return:
    """

    def calc_dist(source, target):
        # what's the range of the distance?
        # this normally is [0, 1]?
        return torch.sqrt(torch.sum((emb[source] - emb[target]) ** 2, dim=1))

    def hop_loss(sample_edge_index):
        # get the average distance between the one hop neighbors
        source, target = sample_edge_index
        dist = calc_dist(source, target)
        return dist.mean(dim=-1)

    logger.info(emb)
    one_hop_loss = hop_loss(edge_index)

    two_hop_edge_index = n_hopway_edge_index_cal(edge_index, emb.shape[0], 2)
    two_hop_loss = hop_loss(two_hop_edge_index)

    three_hop_edge_index = n_hopway_edge_index_cal(edge_index, emb.shape[0], 3)
    three_hop_loss = hop_loss(three_hop_edge_index)
    logger.info(f"one_hop_loss: {one_hop_loss}")
    logger.info(f"two_hop_loss: {two_hop_loss}")
    logger.info(f"three_hop_loss: {three_hop_loss}")
    loss_1 = torch.relu(one_hop_loss * 2 - two_hop_loss)
    loss_2 = torch.relu(one_hop_loss * 3 - three_hop_loss)
    loss_3 = torch.relu(two_hop_loss * 1.5 - three_hop_loss)
    loss = loss_1 + loss_2 + loss_3
    logger.info(f"loss: {loss}")
    return loss


def n_hopway_edge_index_cal(edge_index, num_nodes, n):
    # Convert edge_index to adjacency matrix
    adj = to_dense_adj(edge_index)[0]

    # Compute adjacency matrix to the power of n
    adj_n = torch.matrix_power(adj, n)

    # Apply a mask to set the direct neighbors to zero
    mask = torch.matrix_power(adj, n - 1)
    adj_n[mask > 0] = 0

    # Convert back to edge_index
    n_hopway_edge_index, _ = dense_to_sparse(adj_n)
    logger.info(n_hopway_edge_index.shape)

    # only select part of the edge_index
    n_hopway_edge_index = n_hopway_edge_index[
                          :, n_hopway_edge_index[0] != n_hopway_edge_index[1]
                          ]
    # random select 30 samples from the edge_index
    # n_hopway_edge_index = n_hopway_edge_index[
    #     :, torch.randperm(n_hopway_edge_index.shape[1])[:30]
    # ]

    return n_hopway_edge_index
