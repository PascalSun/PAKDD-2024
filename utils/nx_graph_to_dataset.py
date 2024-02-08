import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx import Graph
from torch_geometric.data import Data

from utils.logger import get_logger

logger = get_logger()


def clean_networkx_node_keys(graph, node_keys=None, edge_keys=None):
    # create a new graph, add the node_keys to the node
    new_graph = Graph()
    if node_keys is not None:
        for node, data in graph.nodes(data=True):
            new_graph.add_node(node, **{key: data[key] for key in node_keys})
    else:
        for node, data in graph.nodes(data=True):
            new_graph.add_node(node, **{})
    if edge_keys is not None:
        for u, v, data in graph.edges(data=True):
            new_graph.add_edge(u, v, **{key: data[key] for key in edge_keys})
    else:
        for u, v, data in graph.edges(data=True):
            new_graph.add_edge(u, v, **{})
    return new_graph


def convert_networkx_graph_to_index(graph):
    # Convert node keys to integer
    index_mapping = {}
    index_graph = nx.Graph()
    node_mapping = {}
    for index, node in enumerate(graph.nodes):
        node_mapping[node] = index
        index_mapping[index] = node
        # Convert node key to integer
        index_graph.add_node(index, **graph.nodes[node])  # Copy node attributes

    for edge in graph.edges:
        node_start = node_mapping[edge[0]]
        node_end = node_mapping[edge[1]]
        index_graph.add_edge(node_start, node_end)  # This line is corrected
    return index_graph, index_mapping


def convert_networkx_to_torch_graph_with_centrality_features(
        nx_graph: Graph,
        degree_df: pd.DataFrame,
        betweenness_df: pd.DataFrame,
        y_field: str = "y",
):
    """
    TODO: this function has some problem, need to meet the standard from geometric package
    Convert networkx graph to pytorch geometric graph, and put the between and degree centrality as features

    It will assume the Graph node already been split into train, val, test
    :param nx_graph:
    :param degree_df:
    :param betweenness_df:
    :param y_field:
    :return:
    """
    edge_index = torch.tensor(list(nx_graph.edges), dtype=torch.long).t().contiguous()

    # Get the number of nodes
    num_nodes = len(nx_graph.nodes())

    # Prepare the feature matrix
    features = np.zeros((num_nodes, 2))
    for node_index, node_key in enumerate(nx_graph.nodes()):
        features[node_index, 0] = degree_df[node_key]
        features[node_index, 1] = betweenness_df[node_key]
    features = torch.tensor(features, dtype=torch.float)

    y = torch.tensor(
        [float(nx_graph.nodes[i].get(y_field, -1)) for i in nx_graph.nodes],
        dtype=torch.long,
    )
    # TODO: this will assume the graph already have the train, val, test mask as the node attribute
    train_mask = torch.tensor(
        [bool(nx_graph.nodes[i].get("train_mask", False)) for i in nx_graph.nodes],
        dtype=torch.bool,
    )
    val_mask = torch.tensor(
        [bool(nx_graph.nodes[i].get("val_mask", False)) for i in nx_graph.nodes],
        dtype=torch.bool,
    )
    test_mask = torch.tensor(
        [bool(nx_graph.nodes[i].get("test_mask", False)) for i in nx_graph.nodes],
        dtype=torch.bool,
    )
    data = Data(x=features, edge_index=edge_index, y=y)

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
