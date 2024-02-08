import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from utils.nx_graph_to_dataset import (
    clean_networkx_node_keys,
    convert_networkx_graph_to_index,
)
from utils.logger import get_logger

logger = get_logger()


class OSMDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super(OSMDataset, self).__init__(root, transform, pre_transform)
        self.name = name
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "/home/pascal/PhD/code/stkg/data/wa/road/openstreetmap/processed/graph.graphml"

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        node_attribute_keys = ["x", "y", "street_count", "target"]
        osm_graph = nx.read_graphml(self.raw_file_names)
        osm_graph = clean_networkx_node_keys(osm_graph, node_attribute_keys)
        osm_graph, _ = convert_networkx_graph_to_index(osm_graph)

        edge_index = torch.tensor(list(osm_graph.edges)).t().contiguous()
        num_nodes = len(osm_graph.nodes())

        # Prepare the feature matrix
        features = np.zeros((num_nodes, len(node_attribute_keys) - 1))
        for node_index, node_key in enumerate(osm_graph.nodes()):
            for i, key in enumerate(node_attribute_keys[0:-1]):
                features[node_index, i] = osm_graph.nodes[node_key][key]

        features = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(
            [float(osm_graph.nodes[i].get("target", 0)) for i in osm_graph.nodes],
            dtype=torch.long,
        )
        data = Data(x=features, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)

        # add train, val, test mask, randomly split 80%, 10%, 10%
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for node_index, node_key in enumerate(osm_graph.nodes()):
            random_number = np.random.random()
            if random_number < 0.8:
                train_mask[node_index] = True
            elif random_number < 0.9:
                val_mask[node_index] = True
            else:
                test_mask[node_index] = True
        data.test_mask = test_mask
        data.train_mask = train_mask
        data.val_mask = val_mask
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

    def len(self):
        return 1

    def get(self, idx):
        return self.get(idx)
