from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from utils.nx_graph_to_dataset import (
    clean_networkx_node_keys,
    convert_networkx_graph_to_index,
)
from utils.logger import get_logger

logger = get_logger()


class MitchamDataset(InMemoryDataset):
    TRAFFIC_COLUMNS = [
        "TRAFFIC COUNT DATA - Combined",
        "TRAFFIC COUNT DATA -Speed_85_c",
        "85th %ile Speed>PSL",
        "COUNTS - AM ",
        "COUNTS - PM",
        "Eastbound",
        "Westbound",
    ]
    RISK_COLUMNS = ["TOTAL CASUALTIES", "TOTAL CRASHES"]

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        self.road_segment_df = None
        self.road_adj_df = None
        self.graph = nx.Graph()
        self.labels = ["Low", "Medium", "High"]
        super(MitchamDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        pass

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_file_names(self):
        return ["mitcham_raw_data.csv", "Road_Adj_Join.csv"]

    def process(self):
        self.road_segment_df = pd.read_csv(
            Path(self.root) / "raw" / self.raw_file_names[0]
        )
        self.road_adj_df = pd.read_csv(Path(self.root) / "raw" / self.raw_file_names[1])
        self.load_road_network()
        self.load_intersection()
        logger.info(f"Constructed graph: {self.graph}")
        mitcham_road_graph = clean_networkx_node_keys(
            self.graph, node_keys=self.TRAFFIC_COLUMNS + ["y"]
        )
        mitcham_road_graph, _ = convert_networkx_graph_to_index(mitcham_road_graph)
        self.graph = mitcham_road_graph

        # get all attributes for the nodes to features
        features = np.zeros((len(self.graph.nodes), len(self.TRAFFIC_COLUMNS)))
        for i, node in enumerate(self.graph.nodes):
            for j, column in enumerate(self.TRAFFIC_COLUMNS):
                features[i, j] = self.graph.nodes[node][column]

        features = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(
            [self.graph.nodes[node]["y"] for node in self.graph.nodes], dtype=torch.long
        )
        edge_index = torch.tensor(list(self.graph.edges)).t().contiguous()
        data = Data(x=features, y=y, edge_index=edge_index)

        num_nodes = len(self.graph.nodes)
        data = data if self.pre_transform is None else self.pre_transform(data)
        # add train, val, test mask, randomly split 80%, 10%, 10%
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for node_index, node_key in enumerate(self.graph.nodes()):
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

    def load_road_network(self):
        """
        - load the linkage for road segments
        - load intersection
        - load the properties of road segments
        """
        road_segments_df = self.road_segment_df
        road_segments_df = road_segments_df.drop_duplicates(subset=["ASSET_ID"])
        road_segments_df = road_segments_df.dropna(subset=["ASSET_ID"])

        road_segments = {}
        sorted_road_segments = {}
        for _, row in road_segments_df.iterrows():
            # aggregate road segments within a road
            road_number_key = str(int(float(row["ROAD_NUMBER"])))
            if road_number_key not in road_segments:
                road_segments[road_number_key] = [row]
            else:
                road_segments[road_number_key].append(row)

        # sort the road segments within a road
        for road_number, road_segments in road_segments.items():
            sorted_road_segments[road_number] = sorted(
                road_segments, key=lambda k: float(k["SEGMENT_NO"])
            )
        # create node
        logger.info("try to create node for road segments")
        for road_number, road_segments in sorted_road_segments.items():
            for _, road_segment in enumerate(road_segments):
                """
                Filter the values of the road segment, only keep the values in the TRAFFIC_COLUMNS
                """
                value_dict = road_segment.to_dict()
                property_dict = {}
                # if value is nan, then replace the value with -1
                for road_segment_value_key, value in value_dict.items():
                    if road_segment_value_key in self.TRAFFIC_COLUMNS:
                        if pd.isnull(value):
                            property_dict[road_segment_value_key] = -1
                        else:
                            property_dict[road_segment_value_key] = value
                # add node to the graph
                property_dict["y"] = self.label_nodes(value_dict)
                self.graph.add_node(road_segment["ASSET_ID"], **property_dict)

        # create link rel
        for road_number, road_segments in sorted_road_segments.items():
            for segment_index, road_segment in enumerate(road_segments):
                if segment_index + 1 < len(road_segments):
                    self.graph.add_edge(
                        road_segment["ASSET_ID"],
                        road_segments[segment_index + 1]["ASSET_ID"],
                    )

    def load_intersection(self):
        road_segments_link_df = self.road_adj_df
        # Add intersections to the graph
        for _, row in road_segments_link_df.iterrows():
            if pd.isnull(row["Segment Asset ID"]):
                continue

            for i in range(1, 6):
                intersecting_road_id = row["Intersecting Road ID {}".format(i)]
                """
                If the intersecting road id is nan, then skip the current row
                """
                if pd.isnull(intersecting_road_id):
                    continue

                """
                Next we will need to do two things:
                - if the source node not in the graph, then add the node to the graph
                - if the target node not in the graph, then add the node to the graph

                Also we need to notice that, the node probably in the format: RD_88_20.1.2,
                However, RD_88_20 is exist in the graph, rather than the RD_88_20.1.2
                """
                source_node = row["Segment Asset ID"]
                target_node = intersecting_road_id
                empty_property_dict = {"y": 0}
                for road_segment_value_key in self.TRAFFIC_COLUMNS:
                    empty_property_dict[road_segment_value_key] = -1
                # we do not have either the source node and the target node in the graph
                if not self.graph.has_node(source_node):
                    source_node = source_node.split(".")[0]
                    if not self.graph.has_node(source_node):
                        self.graph.add_node(source_node, **empty_property_dict)

                if not self.graph.has_node(target_node):
                    target_node = target_node.split(".")[0]
                    if not self.graph.has_node(target_node):
                        self.graph.add_node(target_node, **empty_property_dict)

                if self.graph.has_node(target_node) and self.graph.has_node(
                        source_node
                ):
                    self.graph.add_edge(source_node, target_node)

    @staticmethod
    def label_nodes(node_value_dict: dict) -> int:
        """
        Label the nodes with the given node_label
        :param node_value_dict: the node value dict
        """
        if "TOTAL CASUALTIES" in node_value_dict:
            if node_value_dict["TOTAL CASUALTIES"] > 0:
                return 2  # High
        if "TOTAL CRASHES" in node_value_dict:
            if node_value_dict["TOTAL CRASHES"] > 0:
                return 1  # Medium
        return 0  # Low
