import csv
import math
from collections import Counter
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import powerlaw
from networkx import DiGraph
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from torch_geometric.utils import to_networkx
from sklearn.tree import DecisionTreeClassifier, plot_tree
from dataset.load_datasets import load_dataset
from utils.constants import DataSetEnum
from utils.constants import REPORT_DIR
from utils.logger import get_logger
from utils.timer import timer

HOMO_DATASETS = [
    "PubMed",
    "CiteSeer",
    "Cora",
    "AMAZON_COMPUTERS",
    "AMAZON_PHOTO",
    "NELL",
    "CitationsFull_Cora",
    "CitationsFull_CiteSeer",
    "CitationsFull_PubMed",
    "CitationsFull_Cora_ML",
    "CitationsFull_DBLP",
    "Cora_Full",
    "Coauther_CS",
    "Coauther_Physics",
    "AttributedGraphDataset_Wiki",
    "AttributedGraphDataset_Cora",
    "AttributedGraphDataset_CiteSeer",
    "AttributedGraphDataset_Pubmed",
    "AttributedGraphDataset_BlogCatalog",
    "AttributedGraphDataset_PPI",
    "AttributedGraphDataset_Flickr",
    "AttributedGraphDataset_Facebook",
    "WEBKB_Cornell",
    "WEBKB_Texas",
    "WEBKB_Wisconsin",
    "HeterophilousGraphDataset_Roman_empire",
    "HeterophilousGraphDataset_Amazon_ratings",
    "HeterophilousGraphDataset_Minesweeper",
    "HeterophilousGraphDataset_Tolokers",
    "HeterophilousGraphDataset_Questions",
    "Actor",
    "GitHub",
    "TWITCH_DE",
    "TWITCH_EN",
    "TWITCH_ES",
    "TWITCH_FR",
    "TWITCH_PT",
    "TWITCH_RU",
    "PolBlogs",
    "EllipticBitcoinDataset",
    "DGraphFin",
    "Flickr",
    "Yelp",
    "Reddit",
    # "AMAZON_PRODUCTS",
]

finished_dataset = [
    "AMAZON_COMPUTERS",
    "AMAZON_PHOTO",
    "Actor",
    "AttributedGraphDataset_BlogCatalog",
    "AttributedGraphDataset_CiteSeer",
    "AttributedGraphDataset_Cora",
    "AttributedGraphDataset_Flickr",
    "AttributedGraphDataset_Pubmed",
    "AttributedGraphDataset_Wiki",
    "CitationsFull_CiteSeer",
    "CitationsFull_Cora",
    "CitationsFull_Cora_ML",
    "CitationsFull_DBLP",
    "CitationsFull_PubMed",
    "CiteSeer",
    "Coauther_CS",
    "Coauther_Physics",
    "Cora",
    "Cora_Full",
    "GitHub",
    "HeterophilousGraphDataset_Amazon_ratings",
    "HeterophilousGraphDataset_Minesweeper",
    "HeterophilousGraphDataset_Questions",
    "HeterophilousGraphDataset_Roman_empire",
    "HeterophilousGraphDataset_Tolokers",
    "PubMed",
    "TWITCH_DE",
    "TWITCH_EN",
    "TWITCH_ES",
    "TWITCH_FR",
    "TWITCH_PT",
    "TWITCH_RU",
    "WEBKB_Cornell",
    "WEBKB_Texas",
    "WEBKB_Wisconsin",
]

"""
- how the embedding learned for graph embedding algorithms (random walk, structure/message), node embedding
- dataset network metrics
  with each dataset, domain information, expected characteristics(whether it is dense connected or sparse connected, some citation are important)
  - provide metrics to verify that

"""


class DatasetNetworkMetrics:
    """
    calculate network metrics for a dataset
    for example

    - [x] degree distribution
    - [x] degree exponent ?
    - [x] degree exponent confidence interval
    - [x] number of nodes and edges
    - [x] average degree
    - [x] standard deviation of degree
    - [x] k_min and k_max vs powerlaw calculated k_max
    - [x] average clustering coefficient
    - [x] first moment, which is the average degree vs second moment, which is the variance of the degree
    - [x] goodness of fit for power law
    - [x] distance
        - average shortest path length
        - diameter
    - exponentially bounded or fat tailed network
    -------------------------

    Explain
    -------

    ## Graph metrics distribution
    - dataset: Name of the dataset
    - num_nodes: Number of nodes in the graph
    - num_edges: Number of edges in the graph
    - k: Average degree of the graph, for directed graph, it is the average of in-degree and out-degree
    - ln_n: log of the number of nodes
    - std_k: Standard deviation of the degree distribution
    - gamma_degree: Degree exponent of the degree distribution (power law)
    - gamma_confidence_level: Confidence interval of the degree exponent of the degree distribution (power law)
    - k_min: Minimum degree of the graph
    - k_max: Maximum degree of the graph
    - p_k_max: Power law calculated k_max of the graph (the applied k_min here will be the minimum degree of the graph except 0)
    - k_min_gt0: Minimum degree of the graph except 0
    - k_in_min: Minimum in-degree of the graph
    - k_in_max: Maximum in-degree of the graph
    - p_k_in_max: Power law calculated in-degree k_max of the graph (the applied k_min here will be the minimum in-degree of the graph except 0)
    - k_out_min: Minimum out-degree of the graph
    - k_out_max: Maximum out-degree of the graph
    - p_k_out_max: Power law calculated out-degree k_max of the graph (the applied k_min here will be the minimum out-degree of the graph except 0)
    - d_max_random: Diameter of a graph based on random network model: lnN/ln<k>
    - average_clustering_coefficient: Average clustering coefficient of the graph, measure the local clustering of the graph
    - transitivity: Transitivity of the graph, measure the global clustering of the graph
    - average_shortest_path_length: Average shortest path length of the graph, measure the average distance between two nodes
        - if the graph is disconnected, calculate the average shortest path length of all the largest connected component
    - diameter: Diameter of the graph, measure the maximum distance between two nodes
       - if the graph is disconnected, calculate the diameter of all the largest connected component

    ## Random network model
    Random network model can efficiently explain the small world phenomenon, but not the power law distribution.
    - sub_critical: 0 < <k> < 1 (p<1/N)
    - critical: <k> = 1 (p=1/N)
    - super_critical: 1 < <k> < lnN (1/N < p < lnN/N)
    - connected: <k> > lnN (p > lnN/N)

    Most of our datasets are in either super_critical or connected regime.
    - plot k and ln n is for this purpose
    - plot k and standard deviation is for this purpose

    - d_max_random (diameter) of a graph based on random network model: lnN/ln<k>
        Average Path Length
            Random network theory predicts that the average path length follows (3.19),
            a prediction that offers a reasonable approximation for the observed path lengths.
            Hence the random network model can account for the emergence of small world phenomena.

    ## Clustering coefficient
    - average clustering coefficient
    - transitivity

    ## Scale-free network, which degree distribution follows power law distribution

    ### gamma

    One way to do the distinguish: 2, 2-3, 3, >3, this is use distance to vary the network

    - gamma = 2, network is anomalous regime, average d is constant, no large network can exist here, k2 diverges, k diverges
    - 2 < gamma < 3, network is ultra-small world regime, average d is lnlnN, k finite, k2 diverges
    - gamma = 3, network is at the critical point, average d is lnN / lnlnN
    - gamma > 3, network is small world regime, average d is lnN, k finite, k2 finite

    Another way to do the distinguish: < 2, 2-3, >3

    - < 2, anomalous regime, k diverges, k2 diverges
    - 2-3, scale-free regime, k finite, k2 diverges
    - > 3, random regime, k finite, k2 finite
    """

    def __init__(self, name: str, graph: DiGraph):
        self.graph = graph
        self.dataset_name = name
        self.logger = get_logger()
        self.report_dir = REPORT_DIR / name / "network_metrics"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def exist_datasets(
            update: bool = False, df: Optional[pd.DataFrame] = None
    ) -> Tuple[list, Optional[pd.DataFrame]]:
        if update:
            df.to_csv(
                REPORT_DIR / "summary" / "summary_network_metrics.csv",
                index=False,
            )
            return df["dataset"].tolist(), df
        summary_folder = REPORT_DIR / "summary"
        summary_csv = summary_folder / "summary_network_metrics.csv"
        if not summary_csv.exists():
            with open(summary_csv, "w", newline="") as f:
                writer = csv.writer(f)
                header = [
                    "dataset",
                    "num_nodes",
                    "num_edges",
                    "k",
                    "ln_n",
                    "std_k",
                    "std_k_in",
                    "std_in_out",
                    "k_min",
                    "k_max",
                    "k_in_min",
                    "k_in_max",
                    "k_out_min",
                    "k_out_max",
                    "d_max_random",
                    "average_clustering_coefficient",
                    "transitivity",
                ]
                writer.writerow(header)
        if not summary_csv.exists():
            return [], None
        # read the csv file without index column
        df = pd.read_csv(summary_csv, index_col=False)
        return df["dataset"].tolist(), df

    @staticmethod
    def require_calculations(name: str, df: pd.DataFrame) -> list:
        """
        Check whether the dataset requires calculation
        Returns
        -------

        """
        calculate_list = []
        # if there is no num_node field in the data, then run graph_attribute
        dataset_dict = df.loc[df["dataset"] == name].to_dict(orient="records")
        if len(dataset_dict):
            dataset_dict = dataset_dict[0]
        else:
            dataset_dict = {}

        if "num_nodes" not in dataset_dict:
            calculate_list.append("graph_attribute")
        if "gamma_degree" not in dataset_dict or pd.isna(dataset_dict["gamma_degree"]):
            calculate_list.append("calculate_degree_exponent")
        if "k2" not in dataset_dict or pd.isna(dataset_dict["k2"]):
            calculate_list.append("calculate_second_moment")
        if "gamma_regime" not in dataset_dict or pd.isna(dataset_dict["gamma_regime"]):
            calculate_list.append("calculate_gamma_regime")
        if "p_k_max" not in dataset_dict or pd.isna(dataset_dict["p_k_max"]):
            calculate_list.append("calculate_k_max")
        # calculate average_shortest_path_length
        if "average_shortest_path_length" not in dataset_dict or pd.isna(
                dataset_dict["average_shortest_path_length"]
        ):
            calculate_list.append("calculate_average_shortest_path_length")
        # calculate diameter
        if "diameter" not in dataset_dict or pd.isna(dataset_dict["diameter"]):
            calculate_list.append("calculate_diameter")
        if "num_classes" not in dataset_dict or pd.isna(dataset_dict["num_classes"]):
            calculate_list.append("calculate_num_classes")
        if "num_features" not in dataset_dict or pd.isna(dataset_dict["num_features"]):
            calculate_list.append("calculate_num_features")

        # for reddit and DGraphFin, do not calculate the calculate_average_shortest_path_length and diameter
        if name == "Reddit" or name == "DGraphFin" or name == "Yelp":
            if "calculate_diameter" in calculate_list:
                calculate_list.remove("calculate_diameter")
            if "calculate_average_shortest_path_length" in calculate_list:
                calculate_list.remove("calculate_average_shortest_path_length")
        return calculate_list

    def calculate_graph_attribute(self):
        """
        Log the attributes of the graph
        like the number of nodes, edges, etc.

        Calculate the local coefficient of the graph is hard
        Returns
        -------

        """
        summary_folder = REPORT_DIR / "summary"
        summary_folder.mkdir(parents=True, exist_ok=True)
        summary_csv = summary_folder / "summary_network_metrics.csv"

        if not summary_csv.exists():
            with open(summary_csv, "w", newline="") as f:
                writer = csv.writer(f)
                header = [
                    "dataset",
                    "num_nodes",
                    "num_edges",
                    "k",
                    "ln_n",
                    "std_k",
                    "std_k_in",
                    "std_in_out",
                    "k_min",
                    "k_max",
                    "k_in_min",
                    "k_in_max",
                    "k_out_min",
                    "k_out_max",
                    "d_max_random",
                    "average_clustering_coefficient",
                    "transitivity",
                ]
                writer.writerow(header)

        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        # calculate the average degree
        if self.graph.is_directed():
            k = self.graph.number_of_edges() / self.graph.number_of_nodes()
        else:
            k = 2 * self.graph.number_of_edges() / self.graph.number_of_nodes()

        # calculate the degree distribution
        degree_distribution: dict = self.calculate_degree_distribution()  # noqa

        # calculate the d_max_random (diameter) of a graph based on random network model: lnN/ln<k>
        d_max_random = math.log(num_nodes) / math.log(k)

        # average clustering coefficient and transitivity
        with timer(self.logger, "Calculate average clustering coefficient"):
            average_clustering_coefficient = nx.average_clustering(self.graph)
        with timer(self.logger, r"Calculate transitivity"):
            transitivity = nx.transitivity(self.graph)
        self.logger.info(
            f"Average clustering coefficient: {average_clustering_coefficient}"
        )
        self.logger.info(f"Transitivity: {transitivity}")

        # plot a cluster coefficient distribution
        self.plot_cluster_coefficient_distribution()

        ln_n = math.log(num_nodes)

        # append the result to the file
        # or if the record already exists, then update the record

        # row to df row
        row_data = {
            "dataset": self.dataset_name,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "k": k,
            "ln_n": ln_n,
            "std_k": degree_distribution.get("std_degree"),
            "std_k_in": degree_distribution["std_in_degree"],
            "std_in_out": degree_distribution["std_out_degree"],
            "k_min": degree_distribution["k_min"],
            "k_max": degree_distribution["k_max"],
            "k_in_min": degree_distribution["k_in_min"],
            "k_in_max": degree_distribution["k_in_max"],
            "k_out_min": degree_distribution["k_out_min"],
            "k_out_max": degree_distribution["k_out_max"],
            "d_max_random": d_max_random,
            "average_clustering_coefficient": average_clustering_coefficient,
            "transitivity": transitivity,
            "k_min_gt0": degree_distribution["k_min_gt0"],
            "k_in_min_gt0": degree_distribution["k_in_min_gt0"],
            "k_out_min_gt0": degree_distribution["k_out_min_gt0"],
        }

        dataset_df = pd.read_csv(summary_csv, index_col=False)
        print(dataset_df)
        if self.dataset_name in dataset_df["dataset"].tolist():
            self.logger.info(dataset_df.loc[dataset_df["dataset"] == self.dataset_name])
            for col in row_data:
                dataset_df.loc[dataset_df["dataset"] == self.dataset_name, col] = str(
                    row_data[col]
                )
        else:
            dataset_df = pd.concat(
                [dataset_df, pd.DataFrame([row_data])], ignore_index=True
            )

        dataset_df.to_csv(summary_csv, index=False)
        self.logger.info(f"N/L/<K>lnN: {num_edges}/{num_nodes}/{k}/{ln_n}")

    def calculate_degree_distribution(self) -> Dict:
        """
        Plot the degree distribution

        If it is a directed graph, then we will plot the in-degree and out-degree distribution

        - degree distribution
        - standard deviation of degree
        - degree exponent
        - degree exponent confidence interval
        - k_min and k_max vs powerlaw calculated k_max

        Returns
        -------
            {
                "std_in_degree": std_in_degree,
                "std_out_degree": std_out_degree,
                "std_degree": std_degree,
                "k_in_min": k_in_min,
                "k_in_max": k_in_max,
                "k_out_min": k_out_min,
                "k_out_max": k_out_max,
                "k_min": min(degrees),
                "k_max": max(degrees),
                "k_in_min_gt0": k_in_min_gt0,
                "k_out_min_gt0": k_out_min_gt0,
                "k_min_gt0": k_min_gt0,
            }
        """
        if self.graph.is_directed():
            in_degrees = [self.graph.in_degree(n) for n in self.graph.nodes()]  # noqa
            out_degrees = [self.graph.out_degree(n) for n in self.graph.nodes()]  # noqa
            k_in_min = min(in_degrees)
            k_in_min_gt0 = min([val for val in in_degrees if val > 0])
            k_in_max = max(in_degrees)
            k_out_min = min(out_degrees)
            k_out_min_gt0 = min([val for val in out_degrees if val > 0])
            k_out_max = max(out_degrees)
            in_degrees_distribution = Counter(in_degrees)
            out_degrees_distribution = Counter(out_degrees)
            degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            degrees_distribution = Counter(degrees)
            k_min_gt0 = min([val for val in degrees if val > 0])

            self.plot_degree_distribution(
                in_degrees_distribution, filename="degree_in_distribution.png"
            )
            self.plot_degree_distribution(
                out_degrees_distribution, filename="degree_out_distribution.png"
            )
            self.plot_degree_distribution(
                degrees_distribution, filename="degree_distribution.png"
            )

            std_in_degree = np.std(in_degrees)
            std_out_degree = np.std(out_degrees)
            std_degree = np.std(degrees)

            return {
                "std_in_degree": std_in_degree,
                "std_out_degree": std_out_degree,
                "std_degree": std_degree,
                "k_in_min": k_in_min,
                "k_in_max": k_in_max,
                "k_out_min": k_out_min,
                "k_out_max": k_out_max,
                "k_min": min(degrees),
                "k_max": max(degrees),
                "k_in_min_gt0": k_in_min_gt0,
                "k_out_min_gt0": k_out_min_gt0,
                "k_min_gt0": k_min_gt0,
            }
        else:
            degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            degrees_distribution = Counter(degrees)
            # k_min and k_max
            k_min = min(degrees)
            k_min_gt0 = min([val for val in degrees if val > 0])
            k_max = max(degrees)
            # plot the degree distribution
            self.plot_degree_distribution(
                degrees_distribution, filename="degree_distribution_undirected.png"
            )
            # calculate the standard deviation
            std_degree = np.std(degrees)

            return {
                "std_degree": std_degree,
                "std_in_degree": -1,
                "std_out_degree": -1,
                "k_min": k_min,
                "k_max": k_max,
                "k_in_min": -1,
                "k_in_max": -1,
                "k_out_min": -1,
                "k_out_max": -1,
                "k_in_min_gt0": -1,
                "k_out_min_gt0": -1,
                "k_min_gt0": k_min_gt0,
            }

    def calculate_degree_exponent(self):
        """
        Comments
        It means that each of the nodes with the growth of a node,
        the probability is dependent on the degree of the node.

        However, it will be possible that the probability is not dependent just on the degree of the node,
        And which node will become the high degree nodes, it will depend on its inside characteristics?

        Returns
        -------

        """
        # Fit the degree distribution to a power-law
        degrees = [deg for _, deg in self.graph.degree()]
        fit = powerlaw.Fit(degrees, discrete=True)
        gamma = fit.alpha  # This is the degree exponent

        # the ks test
        ks_statistic = fit.power_law.KS()

        _, df = self.exist_datasets()
        df.loc[df["dataset"] == self.dataset_name, "gamma_degree"] = gamma
        df.loc[df["dataset"] == self.dataset_name, "ks_statistic"] = ks_statistic
        self.exist_datasets(update=True, df=df)

        # do the same thing for in-degree and out-degree
        if self.graph.is_directed():
            degrees_in = [deg for _, deg in self.graph.in_degree()]  # noqa
            fit_in = powerlaw.Fit(degrees_in, discrete=True)
            gamma_in = fit_in.alpha  # This is the degree exponent
            ks_statistic_in = fit_in.power_law.KS()

            degrees_out = [deg for _, deg in self.graph.out_degree()]  # noqa
            fit_out = powerlaw.Fit(degrees_out, discrete=True)
            gamma_out = fit_out.alpha  # This is the degree exponent
            ks_statistic_out = fit_out.power_law.KS()

            _, df = self.exist_datasets()
            df.loc[df["dataset"] == self.dataset_name, "gamma_in_degree"] = gamma_in
            df.loc[df["dataset"] == self.dataset_name, "gamma_out_degree"] = gamma_out
            df.loc[
                df["dataset"] == self.dataset_name, "ks_statistic_in"
            ] = ks_statistic_in
            df.loc[
                df["dataset"] == self.dataset_name, "ks_statistic_out"
            ] = ks_statistic_out

        else:
            _, df = self.exist_datasets()
            df.loc[df["dataset"] == self.dataset_name, "gamma_in_degree"] = -1
            df.loc[df["dataset"] == self.dataset_name, "gamma_out_degree"] = -1
            df.loc[df["dataset"] == self.dataset_name, "ks_statistic_in"] = -1
            df.loc[df["dataset"] == self.dataset_name, "ks_statistic_out"] = -1
        self.exist_datasets(update=True, df=df)

    def calculate_gamma_regime(self):
        """
        Calculate the gamma regime of the graph based on the degree exponent
        Returns
        -------

        """
        _, df = self.exist_datasets()

        df.loc[
            df["dataset"] == self.dataset_name, "gamma_regime"
        ] = self.calculate_degree_exponent_regime(
            df.loc[df["dataset"] == self.dataset_name, "gamma_degree"].values[0]
        )
        self.exist_datasets(update=True, df=df)

    def calculate_second_moment(self):
        """
        Calculate the second moment of a graph G.

        Parameters:
        - G: A networkx graph.

        Returns:
        - The second moment of the graph.
        """
        _, df = self.exist_datasets()
        # Get the degrees of all nodes
        degrees = [deg for _, deg in self.graph.degree()]

        # Calculate the second moment
        second_moment = sum(k ** 2 for k in degrees) / len(degrees)
        # df.loc[df["dataset"] == self.dataset_name, "k2"] = second_moment
        matching_rows = df[df["dataset"] == self.dataset_name]
        if len(matching_rows) == 1:
            df.loc[df["dataset"] == self.dataset_name, "k2"] = second_moment
        else:
            # Handle the case where the assumption does not hold, maybe raise an error or log a warning
            raise ValueError("Expected exactly one matching row, found {}".format(len(matching_rows)))

        self.exist_datasets(update=True, df=df)

    def calculate_p_k_max(self):
        _, df = self.exist_datasets()
        k_min_gt0 = df.loc[df["dataset"] == self.dataset_name, "k_min_gt0"].values[0]
        num_nodes = df.loc[df["dataset"] == self.dataset_name, "num_nodes"].values[0]
        gamma = df.loc[df["dataset"] == self.dataset_name, "gamma_degree"].values[0]
        p_k_max = k_min_gt0 * (num_nodes ** (1 / (gamma - 1)))

        if self.graph.is_directed():
            k_in_min_gt0 = df.loc[
                df["dataset"] == self.dataset_name, "k_in_min_gt0"
            ].values[0]
            k_out_min_gt0 = df.loc[
                df["dataset"] == self.dataset_name, "k_out_min_gt0"
            ].values[0]
            gamma_in = df.loc[
                df["dataset"] == self.dataset_name, "gamma_in_degree"
            ].values[0]
            gamma_out = df.loc[
                df["dataset"] == self.dataset_name, "gamma_out_degree"
            ].values[0]
            p_k_in_max = k_in_min_gt0 * (num_nodes ** (1 / (gamma_in - 1)))
            p_k_out_max = k_out_min_gt0 * (num_nodes ** (1 / (gamma_out - 1)))

        else:
            p_k_in_max = -1
            p_k_out_max = -1

        df.loc[df["dataset"] == self.dataset_name, "p_k_max"] = p_k_max
        df.loc[df["dataset"] == self.dataset_name, "p_k_in_max"] = p_k_in_max
        df.loc[df["dataset"] == self.dataset_name, "p_k_out_max"] = p_k_out_max
        self.exist_datasets(update=True, df=df)

    @staticmethod
    def calculate_degree_exponent_regime(gamma):
        if gamma <= 2:
            return "anomalous"
        elif gamma < 3:
            return "scale-free-ultra-small-world"
        elif gamma == 3:
            return "critical-point"
        else:
            return "random-small-world"

    @staticmethod
    def compute_average_path_single(graph, nodes):
        # Extract the subgraph from the graph using the nodes of the connected component
        subgraph = graph.subgraph(nodes)
        return nx.average_shortest_path_length(subgraph)

    def calculate_average_shortest_path_length(self):
        """
        Calculate the average shortest path length of a graph G.
        Returns
        -------
        """
        _, df = self.exist_datasets()
        # get the graph to be an undirected graph
        self.logger.info(f"Is the graph directed: {self.graph.is_directed()}")
        if self.graph.is_directed():
            graph = self.graph.to_undirected()
        else:
            graph = self.graph
        self.logger.info(f"Is the graph connected: {nx.is_connected(graph)}")
        if nx.is_connected(graph):
            average_shortest_path_length = nx.average_shortest_path_length(graph)
            print("-----------------------------")
            print(average_shortest_path_length)
        else:
            # find the average shortest path length of each connected component
            # average_shortest_path_length = np.mean(
            #     [
            #         nx.average_shortest_path_length(graph.subgraph(c))
            #         for c in nx.connected_components(graph)
            #     ]
            # )
            with Pool(cpu_count()) as pool:
                # Compute the average shortest path length for each connected component in parallel
                # Use partial to fix the graph argument for compute_average_path
                func = partial(self.compute_average_path_single, graph)
                average_shortest_path_lengths = pool.map(
                    func, nx.connected_components(graph)
                )
                print(average_shortest_path_lengths)
                # Get the maximum average shortest path length
                average_shortest_path_length = max(average_shortest_path_lengths)
        df.loc[
            df["dataset"] == self.dataset_name, "average_shortest_path_length"
        ] = average_shortest_path_length
        self.exist_datasets(update=True, df=df)

    @staticmethod
    def compute_diameter_single(graph, nodes):
        # Extract the subgraph from the graph using the nodes of the connected component
        subgraph = graph.subgraph(nodes)
        return nx.diameter(subgraph)

    def calculate_diameter(self):
        """
        Calculate the diameter of a graph G.
        Returns
        -------

        """
        _, df = self.exist_datasets()
        if self.graph.is_directed():
            graph = self.graph.to_undirected()
        else:
            graph = self.graph
        if nx.is_connected(graph):
            diameter = nx.diameter(graph)
        else:
            # find the max diameter of each connected component
            # single process version
            # diameter = max(
            #     [nx.diameter(graph.subgraph(c)) for c in nx.connected_components(graph)]
            # )
            with Pool(cpu_count()) as pool:
                # Compute the diameter for each connected component in parallel
                # Use partial to fix the graph argument for compute_diameter
                func = partial(self.compute_diameter_single, graph)
                diameters = pool.map(func, nx.connected_components(graph))
                # Get the maximum diameter
                diameter = max(diameters)

        df.loc[df["dataset"] == self.dataset_name, "diameter"] = diameter
        self.exist_datasets(update=True, df=df)

    def calculate_num_classes(self, dataset):
        _, df = self.exist_datasets()
        df.loc[df["dataset"] == self.dataset_name, "num_classes"] = dataset.num_classes
        self.exist_datasets(update=True, df=df)

    def calculate_num_features(self, dataset):
        _, df = self.exist_datasets()
        df.loc[
            df["dataset"] == self.dataset_name, "num_features"
        ] = dataset.num_features
        self.exist_datasets(update=True, df=df)

    def plot_degree_distribution(
            self, distribution, filename: str = "degree_distribution.png"
    ):
        """
        For the degree distribution, we will plot the degree and the count

        And then plot the loglog distribution, we should find out it follow the power law distribution

        Parameters
        ----------
        distribution
        filename

        Returns
        -------

        """
        degrees, counts = zip(*sorted(distribution.items()))
        plt.figure(figsize=(10, 5))
        # do it with points
        plt.scatter(degrees, counts)
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.title(f"{self.dataset_name} Degree distribution")
        # save the figure
        plt.savefig(self.report_dir / filename)
        plt.close()
        # plt a log-log plot
        plt.figure(figsize=(10, 5))
        # plot loglog with scatter
        plt.loglog(degrees, counts, "o")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.title(f"{self.dataset_name} Degree distribution (log-log)")
        # save the figure
        plt.savefig(self.report_dir / f"log_{filename}")
        plt.close()

    def plot_cluster_coefficient_distribution(
            self, filename: str = "cluster_coefficient_distribution.png"
    ):
        """
        plot the cluster coefficient distribution for the graph vs the k

        so x_axis will be the k, y_axis will be the cluster coefficient

        do the scatter plot
        """
        # calculate the cluster coefficient for each node
        cluster_coefficient = nx.clustering(self.graph)
        # calculate the degree for each node
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        # plot the scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(degrees, cluster_coefficient.values())
        plt.xlabel("Degree")
        plt.ylabel("Cluster coefficient")
        plt.title(f"{self.dataset_name} Cluster coefficient distribution")
        # save the figure
        plt.savefig(self.report_dir / filename)
        plt.close()
        # plot the log-log plot
        plt.figure(figsize=(10, 5))
        plt.loglog(degrees, cluster_coefficient.values(), "o")
        plt.xlabel("Degree")
        plt.ylabel("Cluster coefficient")
        plt.title(f"{self.dataset_name} Cluster coefficient distribution (log-log)")
        # save the figure
        plt.savefig(self.report_dir / f"log_{filename}")
        plt.close()

    @staticmethod
    def plot_summary_k_and_ln_n_random_network() -> list:
        """
        Plot the k and ln n in summary_network_metrics.csv using Plotly with hover text

        Explanation
        -----------
        This is from the random network model.

        For the
        - 0 < <k> < 1 (p<1/N): sub_critical regime
        - <k> = 1 (p=1/N): critical point
        - 1 < <k> < lnN (1/N < p < lnN/N): super_critical regime
        - <k> > lnN (p > lnN/N): connected regime

        Returns
        -------
        """
        summary_folder = REPORT_DIR / "summary"
        k_ln_n_summary_folder = summary_folder / "k_ln"
        k_ln_n_summary_folder.mkdir(parents=True, exist_ok=True)
        summary_csv = summary_folder / "summary_network_metrics.csv"
        if not summary_csv.exists():
            raise ValueError(f"Summary csv {summary_csv} does not exist")
        df = pd.read_csv(summary_csv, index_col=False)

        # calculate the random network regime for each dataset
        df["ln_n"] = df.apply(lambda row: math.log(row["num_nodes"]), axis=1)
        df["regime"] = df.apply(
            lambda row: "sub_critical"
            if row["k"] < 1
            else "critical"
            if row["k"] == 1
            else "super_critical"
            if row["k"] < row["ln_n"]
            else "connected",
            axis=1,
        )

        # Separate data based on the condition x > 10 and x <= 10
        df_gt_10 = df[df["k"] > 10]
        df_le_10 = df[df["k"] <= 10]

        # Function to create a figure based on a dataframe
        def create_figure(dataframe, create_image: bool = False):
            """
            label the point color based on the regime value
            Parameters
            ----------
            dataframe
            create_image

            Returns
            -------

            """
            fig = go.Figure()
            # Define a color map for the regimes
            color_map = {
                "sub_critical": "gray",
                "critical": "orange",
                "super_critical": "yellow",
                "connected": "blue",
            }

            # Map the regime values in the dataframe to their respective colors
            colors = dataframe["regime"].map(color_map)
            fig.add_trace(
                go.Scatter(
                    x=dataframe["k"],
                    y=dataframe["ln_n"],
                    mode="markers",
                    text=dataframe["dataset"],
                    hoverinfo="text+x+y",
                    marker=dict(size=10, opacity=0.5, color=colors),
                    name="<K> vs lnN",
                )
            )

            # Add lines
            fig.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=1,
                    x1=1,
                    y0=0,
                    y1=max(dataframe["ln_n"]),
                    line=dict(color="red", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[1.5, 16],
                    y=[1.5, 16],
                    mode="lines",
                    line=dict(color="green", dash="dash"),
                    name="lnN = <K>",
                )
            )

            if create_image:
                # Add annotations with reduced font size
                for i in range(len(dataframe)):
                    fig.add_annotation(
                        go.layout.Annotation(
                            x=dataframe["k"].iloc[i],
                            y=dataframe["ln_n"].iloc[i],
                            xref="x",
                            yref="y",
                            text=dataframe["dataset"].iloc[i],
                            showarrow=False,
                            font=dict(size=6),  # Reduced font size
                        )
                    )

            # Update layout
            fig.update_layout(
                title="<K> vs lnN (Average degree vs number of nodes, random network model)",
                xaxis_title="<K>",
                yaxis_title="lnN",
                xaxis=dict(zeroline=False, showgrid=True),
                yaxis=dict(zeroline=False, showgrid=True),
            )
            return fig

        # Create figures
        fig_gt_10 = create_figure(df_gt_10, True)
        fig_le_10 = create_figure(df_le_10, True)

        # Save the figures
        fig_gt_10.write_image(
            str(k_ln_n_summary_folder / "k_gt_10_vs_ln_n.png"),
            width=3200,
            height=600,
            scale=5,
        )
        fig_le_10.write_image(
            str(k_ln_n_summary_folder / "k_le_10_vs_ln_n.png"),
            width=1200,
            height=600,
            scale=5,
        )

        # Save the combined figure as HTML
        combined_fig = create_figure(df)

        combined_fig.write_html(str(k_ln_n_summary_folder / "k_vs_ln_n.html"))

        # Return the list of datasets
        return df["dataset"].tolist()

    @staticmethod
    def plot_summary_k_and_std_random_network():
        """
        plot k and standard deviation, and also in the plot, have a dashed line about the standard_deviation = ⟨k⟩^(1/2)
        """
        summary_folder = REPORT_DIR / "summary"
        summary_csv = summary_folder / "summary_network_metrics.csv"
        k_ln_n_summary_folder = summary_folder / "k_ln"
        k_ln_n_summary_folder.mkdir(parents=True, exist_ok=True)
        if not summary_csv.exists():
            raise ValueError(f"Summary csv {summary_csv} does not exist")

        df = pd.read_csv(summary_csv, index_col=False)

        # plot the k and standard deviation
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["k"],
                y=df["std_k"],
                mode="markers",
                text=df["dataset"],
                hoverinfo="text+x+y",
                marker=dict(size=10, opacity=0.5),
                name="Standard deviation vs <K>",
            )
        )
        # Add a function line σ = ‹k›1/2
        fig.add_trace(
            go.Scatter(
                x=list(range(1, 500)),
                y=[val ** 0.5 for val in list(range(1, 500))],
                mode="lines",
                line=dict(color="green", dash="dash"),
                name="σ = ‹k›1/2",
            )
        )
        # Update layout
        fig.update_layout(
            title="Standard deviation vs <K>",
            xaxis_title="<K>",
            yaxis_title="Standard deviation",
            xaxis=dict(zeroline=False, showgrid=True),
            yaxis=dict(zeroline=False, showgrid=True),
        )
        fig.write_image(
            str(k_ln_n_summary_folder / "k_vs_std.png"), width=1200, height=600, scale=5
        )
        # Save the combined figure as HTML
        fig.write_html(str(k_ln_n_summary_folder / "k_vs_std.html"))
        # close the figures
        plt.close()

    @staticmethod
    def plot_summary_tsne():
        """
        TSNE of the summary_network_metrics.csv

        Returns
        -------

        """

        summary_folder = REPORT_DIR / "summary"
        summary_csv = summary_folder / "summary_network_metrics.csv"
        summary_csv_scaled = summary_folder / "summary_network_metrics_min_max.csv"
        tsne_summary_folder = summary_folder / "tsne"
        tsne_summary_folder.mkdir(parents=True, exist_ok=True)
        if not summary_csv.exists():
            raise ValueError(f"Summary csv {summary_csv} does not exist")

        df = pd.read_csv(summary_csv, index_col=False)
        features = df.drop(columns=["dataset"])
        logger.info(f"Features: {features.columns}")
        # except columns:
        """
        - k_in_min
        - k_in_max
        - k_out_min
        - k_out_max
        - gamma_in_degree
        - gamma_out_degree
        - ks_statistic_in
        - ks_statistic_out
        - k_in_min_gt0
        - k_out_min_gt0
        - p_k_in_max
        - p_k_out_max
        - gamma_regime
        # this will depends, we will construct a tsne with these two columns
        - average_shortest_path_length
        - diameter

        """
        features_degrees = features.drop(
            columns=[
                "k_in_min",
                "k_in_max",
                "k_out_min",
                "k_out_max",
                "gamma_in_degree",
                "gamma_out_degree",
                "ks_statistic_in",
                "ks_statistic_out",
                "k_in_min_gt0",
                "k_out_min_gt0",
                "p_k_in_max",
                "p_k_out_max",
                "gamma_regime",
                "average_shortest_path_length",
                "diameter",
            ]
        )

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features_degrees)
        labels = df["dataset"]
        tsne_df = pd.DataFrame(
            data={"x": tsne_results[:, 0], "y": tsne_results[:, 1], "dataset": labels}
        )

        # Plot using Plotly
        fig = px.scatter(tsne_df, x="x", y="y", color="dataset")
        # fig.show()
        fig.write_html(str(tsne_summary_folder / "tsne.html"))

        # Plot threeD
        tsne = TSNE(n_components=3, random_state=42)
        tsne_results = tsne.fit_transform(features_degrees)
        labels = df["dataset"]
        tsne_df = pd.DataFrame(
            data={
                "x": tsne_results[:, 0],
                "y": tsne_results[:, 1],
                "z": tsne_results[:, 2],
                "dataset": labels,
            }
        )

        fig = px.scatter_3d(tsne_df, x="x", y="y", z="z", color="dataset")
        # fig.show()
        fig.write_html(str(tsne_summary_folder / "tsne_3d.html"))

        # plot the tsne with the average shortest path length
        # read the summary csv again
        df = pd.read_csv(summary_csv, index_col=False)
        # drop na for the average shortest path length and diameter
        df = df.dropna(subset=["average_shortest_path_length", "diameter"])
        features = df.drop(columns=["dataset"])
        logger.debug(f"Features: {features.columns}")
        # except columns:
        features_distances = features.drop(
            columns=[
                "k_in_min",
                "k_in_max",
                "k_out_min",
                "k_out_max",
                "gamma_in_degree",
                "gamma_out_degree",
                "ks_statistic_in",
                "ks_statistic_out",
                "k_in_min_gt0",
                "k_out_min_gt0",
                "p_k_in_max",
                "p_k_out_max",
                "gamma_regime",
            ]
        )
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features_distances)
        labels = df["dataset"]
        tsne_df = pd.DataFrame(
            data={"x": tsne_results[:, 0], "y": tsne_results[:, 1], "dataset": labels}
        )

        # Plot using Plotly
        fig = px.scatter(tsne_df, x="x", y="y", color="dataset")
        # fig.show()
        fig.write_html(str(tsne_summary_folder / "tsne_distances.html"))

        # Plot threeD
        tsne = TSNE(n_components=3, random_state=42)
        tsne_results = tsne.fit_transform(features_distances)
        labels = df["dataset"]
        tsne_df = pd.DataFrame(
            data={
                "x": tsne_results[:, 0],
                "y": tsne_results[:, 1],
                "z": tsne_results[:, 2],
                "dataset": labels,
            }
        )

        fig = px.scatter_3d(tsne_df, x="x", y="y", z="z", color="dataset")
        # fig.show()
        fig.write_html(str(tsne_summary_folder / "tsne_distances_3d.html"))

        # plot tsne with these row combinations
        row_columns_combinations = [
            (
                "num_nodes",
                "num_edges",
                "num_classes",
                "num_features",
                "k_min",
                "k_max",
                "k",
                "k2",
                "ln_average_shortest_path_length",
                "average_clustering_coefficient",
                "transitivity",
                "diameter",
                "gamma_degree",
            ),
            ("ln_num_edges", "ln_num_classes", "ln_average_shortest_path_length"),
            # ("ln_num_classes", "ln_n", "ln_k_max"),
            # ("ln_num_classes", "ln_n", "ln_average_shortest_path_length"),
            # (
            #     "ln_num_classes",
            #     "ln_n",
            #     "ln_k_max",
            #     "ln_average_shortest_path_length",
            # ),
            # (
            #     "ln_num_classes",
            #     "ln_n",
            #     "ln_average_shortest_path_length",
            #     "ln_diameter",
            # ),

        ]

        def convert_rgb(normalized_rgb):
            # Convert each component to the standard 0-255 range
            standard_rgb = tuple(int(x * 255) for x in normalized_rgb)
            # Format as an RGB string
            return f"rgb({standard_rgb[0]}, {standard_rgb[1]}, {standard_rgb[2]})"

        # Define colors
        CoexistenceColor = convert_rgb((0.0, 0.6, 0.8))  # Light blue
        CitationColor = convert_rgb((0.6, 0.3, 0.9))  # Light purple
        SocialColor = convert_rgb((0.3, 0.9, 0.3))  # Light green
        KnowledgeColor = convert_rgb((0.9, 0.2, 0.2))  # Light red
        SocialKnowledgeColor = convert_rgb((0.95, 0.9, 0.2))  # Light yellow
        GridColor = convert_rgb((0.2, 0.7, 0.9))  # Very light blue

        dataset_color_mapping = {
            "Actor": CoexistenceColor,
            "AMAZON_COMPUTERS": CoexistenceColor,
            "AMAZON_PHOTO": CoexistenceColor,
            "AttributedGraphDataset_BlogCatalog": SocialColor,
            "AttributedGraphDataset_CiteSeer": CitationColor,
            "AttributedGraphDataset_Cora": CitationColor,
            "AttributedGraphDataset_Flickr": SocialKnowledgeColor,
            "AttributedGraphDataset_Pubmed": CitationColor,
            "AttributedGraphDataset_Wiki": KnowledgeColor,
            "CitationsFull_CiteSeer": CitationColor,
            "CitationsFull_Cora": CitationColor,
            "CitationsFull_Cora_ML": CitationColor,
            "CitationsFull_DBLP": CitationColor,
            "CitationsFull_PubMed": CitationColor,
            "CiteSeer": CitationColor,
            "Coauther_CS": CoexistenceColor,
            "Coauther_Physics": CoexistenceColor,
            "Cora": CitationColor,
            "Cora_Full": CitationColor,
            "GitHub": SocialColor,
            "HeterophilousGraphDataset_Amazon_ratings": CoexistenceColor,
            "HeterophilousGraphDataset_Minesweeper": GridColor,
            "HeterophilousGraphDataset_Questions": SocialKnowledgeColor,
            "HeterophilousGraphDataset_Roman_empire": KnowledgeColor,
            "HeterophilousGraphDataset_Tolokers": CoexistenceColor,
            "PubMed": CitationColor,
            "TWITCH_DE": SocialColor,
            "TWITCH_EN": SocialColor,
            "TWITCH_ES": SocialColor,
            "TWITCH_FR": SocialColor,
            "TWITCH_PT": SocialColor,
            "TWITCH_RU": SocialColor,
            "WEBKB_Cornell": CitationColor,
            "WEBKB_Texas": CitationColor,
            "WEBKB_Wisconsin": CitationColor,
        }

        dataset_classification_mapping = {
            "Actor": 'cross',
            "AMAZON_COMPUTERS": 'cross',
            "AMAZON_PHOTO": 'cross',
            "AttributedGraphDataset_BlogCatalog": 'cross',
            "AttributedGraphDataset_CiteSeer": 'circle',
            "AttributedGraphDataset_Cora": 'circle',
            "AttributedGraphDataset_Flickr": 'cross',
            "AttributedGraphDataset_Pubmed": 'cross',
            "AttributedGraphDataset_Wiki": 'cross',
            "CitationsFull_CiteSeer": 'circle',
            "CitationsFull_Cora": 'circle',
            "CitationsFull_Cora_ML": 'circle',
            "CitationsFull_DBLP": 'circle',
            "CitationsFull_PubMed": 'cross',
            "CiteSeer": 'circle',
            "Coauther_CS": 'circle',
            "Coauther_Physics": 'circle',
            "Cora": 'circle',
            "Cora_Full": 'circle',
            "GitHub": 'cross',
            "HeterophilousGraphDataset_Amazon_ratings": 'cross',
            "HeterophilousGraphDataset_Minesweeper": 'circle',
            "HeterophilousGraphDataset_Questions": 'circle',
            "HeterophilousGraphDataset_Roman_empire": 'circle',
            "HeterophilousGraphDataset_Tolokers": 'circle',
            "PubMed": 'cross',
            "TWITCH_DE": 'circle',
            "TWITCH_EN": 'circle',
            "TWITCH_ES": 'circle',
            "TWITCH_FR": 'circle',
            "TWITCH_PT": 'circle',
            "TWITCH_RU": 'circle',
            "WEBKB_Cornell": 'circle',
            "WEBKB_Texas": 'circle',
            "WEBKB_Wisconsin": 'circle',
        }
        dataset_node2vec_mapping = {
            "Actor": 'cross',
            "AMAZON_COMPUTERS": 'square',
            "AMAZON_PHOTO": 'square',
            "AttributedGraphDataset_BlogCatalog": 'circle-dot',
            "AttributedGraphDataset_CiteSeer": 'circle-dot',
            "AttributedGraphDataset_Cora": 'circle-dot',
            "AttributedGraphDataset_Flickr": 'circle-dot',
            "AttributedGraphDataset_Pubmed": 'circle-dot',
            "AttributedGraphDataset_Wiki": 'circle-dot',
            "CitationsFull_CiteSeer": 'circle-dot',
            "CitationsFull_Cora": 'square',
            "CitationsFull_Cora_ML": 'square',
            "CitationsFull_DBLP": 'square',
            "CitationsFull_PubMed": 'circle-dot',
            "CiteSeer": 'circle-dot',
            "Coauther_CS": 'square',
            "Coauther_Physics": 'square',
            "Cora": 'square',
            "Cora_Full": 'square',
            "GitHub": 'square',
            "HeterophilousGraphDataset_Amazon_ratings": 'circle-dot',
            "HeterophilousGraphDataset_Minesweeper": 'cross',
            "HeterophilousGraphDataset_Questions": 'circle-dot',
            "HeterophilousGraphDataset_Roman_empire": 'cross',
            "HeterophilousGraphDataset_Tolokers": 'square',
            "PubMed": 'circle-dot',
            "TWITCH_DE": 'cross',
            "TWITCH_EN": 'cross',
            "TWITCH_ES": 'cross',
            "TWITCH_FR": 'cross',
            "TWITCH_PT": 'cross',
            "TWITCH_RU": 'cross',
            "WEBKB_Cornell": 'circle-dot',
            "WEBKB_Texas": 'circle-dot',
            "WEBKB_Wisconsin": 'circle-dot',
        }
        fig = make_subplots(rows=1, cols=len(row_columns_combinations),
                            specs=[[{"type": "scatter"}, {"type": "scatter3d"}]],
                            subplot_titles=["All statistics", "$|E|, Cl, L$"])

        for i, row_columns_combination in enumerate(row_columns_combinations, start=1):
            df = pd.read_csv(summary_csv_scaled, index_col=False)
            df = df.dropna(subset=["average_shortest_path_length", "diameter"])
            df = df[df["dataset"].isin(finished_dataset)]
            logger.info(row_columns_combination)
            logger.info(f"shape: {df.shape}")

            features = df[list(row_columns_combination)]
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(features)
            df['color'] = df['dataset'].map(dataset_color_mapping)
            df['symbol'] = df['dataset'].map(dataset_classification_mapping)
            logger.info(f"color: {df['color']}")

            tsne_df = pd.DataFrame({
                "x": tsne_results[:, 0],
                "y": tsne_results[:, 1],
                "color": df["color"],
                "symbol": df["symbol"],
            })

            if i == 1:

                # Add scatter plot to the subplot
                fig.add_trace(
                    go.Scatter(
                        x=tsne_df["x"],
                        y=tsne_df["y"],
                        mode="markers",
                        text=df["dataset"],
                        hoverinfo="text+x+y",
                        marker=dict(size=10, color=tsne_df["color"], symbol=tsne_df["symbol"]),
                    ),
                    row=1, col=i
                )
            else:
                # Add scatter plot to the subplot
                fig.add_trace(
                    go.Scatter3d(
                        x=df["ln_num_edges"],
                        y=df["ln_num_classes"],
                        z=df["ln_average_shortest_path_length"],
                        mode="markers",
                        text=df["dataset"],
                        hoverinfo="text+x+y+z",  # Include z in hoverinfo
                        marker=dict(size=10, color=df["color"], symbol=df["symbol"])  # Removed the symbol attribute
                    ),
                    row=1, col=i
                )
                print(df[['ln_num_edges', 'ln_num_classes', 'ln_average_shortest_path_length', 'symbol']])

            for j in range(1, i + 1):
                fig.update_xaxes(showticklabels=False, row=1, col=j)
                fig.update_yaxes(showticklabels=False, row=1, col=j)
        # Update layout if needed
        fig.update_layout(showlegend=False, margin=dict(l=25, r=25, t=25, b=25))

        # Save the figure as HTML and PDF
        html_file = tsne_summary_folder / "combined_tsne.html"
        fig.write_html(str(html_file))

        pdf_file = tsne_summary_folder / "combined_tsne.pdf"
        fig.write_image(str(pdf_file), width=400 * len(row_columns_combinations), height=300)

        logger.info("Combined Plot saved")
        for row_columns_combination in row_columns_combinations:
            df = pd.read_csv(summary_csv_scaled, index_col=False)
            # drop na for the average shortest path length and diameter
            df = df.dropna(subset=["average_shortest_path_length", "diameter"])
            # dataset is in the finished dataset
            df = df[df["dataset"].isin(finished_dataset)]
            logger.info(row_columns_combination)
            logger.info(f"shape: {df.shape}")
            features = df.drop(columns=["dataset"])
            logger.debug(f"Features: {features.columns}")
            # only include the columns in the row_columns_combination
            features = features[list(row_columns_combination)]
            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(features)

            df['color'] = df['dataset'].map(dataset_color_mapping)
            df['symbol'] = df['dataset'].map(dataset_classification_mapping)
            tsne_df = pd.DataFrame(
                data={
                    "x": tsne_results[:, 0],
                    "y": tsne_results[:, 1],
                    "dataset": df["dataset"],
                    "color": df["color"],
                    "symbol": df["symbol"]
                }
            )

            # Plot using Plotly
            # fig = px.scatter(tsne_df, x="x", y="y", color="color")
            # fig.update_traces(marker=dict(symbol=tsne_df['symbol']))
            fig = go.Figure()

            # Add a scatter plot trace with custom symbols
            fig.add_trace(
                go.Scatter(
                    x=tsne_df["x"],
                    y=tsne_df["y"],
                    mode='markers',
                    marker=dict(
                        color=tsne_df["color"],
                        symbol=tsne_df["symbol"],  # Set the symbol for each data point
                        size=10
                    )
                )
            )

            fig.update_layout(showlegend=False)
            # Save the figure as HTML
            html_file = tsne_summary_folder / f"tsne_{'_'.join(row_columns_combination)}.html"
            fig.write_html(str(html_file))

            # Save the figure as PDF
            pdf_file = tsne_summary_folder / f"tsne_{'_'.join(row_columns_combination)}.pdf"
            fig.write_image(str(pdf_file), width=400, height=300, scale=10)

    @staticmethod
    def explore_metric_distribution():
        # read the summary csv again
        summary_folder = REPORT_DIR / "summary"
        summary_csv = summary_folder / "summary_network_metrics.csv"
        if not summary_csv.exists():
            raise ValueError(f"Summary csv {summary_csv} does not exist")
        summary_metrics_distribution_folder = summary_folder / "metrics_distribution"
        summary_metrics_distribution_folder.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(summary_csv, index_col=False)
        # remove Reddit, DGraphFin and Yelp
        df = df[df["dataset"] != "Reddit"]
        df = df[df["dataset"] != "DGraphFin"]
        df = df[df["dataset"] != "Yelp"]
        # remove HeterophilousGraphDataset_Roman_empire TODO： explore this later
        # df = df[df["dataset"] != "HeterophilousGraphDataset_Roman_empire"]
        # remove NELL
        df = df[df["dataset"] != "NELL"]

        # the dataset should be within finished dataset
        df = df[df["dataset"].isin(finished_dataset)]

        logger.debug(f"Features: {df.columns}")

        # the concerned columns
        concerned_columns = [
            "num_nodes",
            "num_edges",
            "num_features",
            "num_classes",
            "ln_n",
            "k",
            "std_k",
            "k2",
            "k_min",
            "k_max",
            "d_max_random",
            "gamma_degree",
            "average_shortest_path_length",
            "diameter",
            "average_clustering_coefficient",
            "transitivity",
            # p_k_max
            # ks_statistic
        ]

        def plot_distribution_metrics(columns, plot_df, folder, filename):
            # plot the distribution for each column with plotly in same figure, do it with scatter plot
            # transpose the x and y axis
            # get the figure larger, output in png
            fig = make_subplots(
                rows=4,
                cols=4,
                subplot_titles=columns,
                vertical_spacing=0.1,
                horizontal_spacing=0.1,
            )
            # add the scatter plot for each column
            # also do not give the text for x-axis
            for i, at_column in enumerate(columns):
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["dataset"],
                        y=plot_df[at_column],
                        mode="markers",
                        text=plot_df["dataset"],
                        hoverinfo="text+x+y",
                        marker=dict(size=10, opacity=0.5),
                        name=at_column,
                    ),
                    row=i // 4 + 1,
                    col=i % 4 + 1,
                )
            # update the x-axis, make it show nothing
            fig.update_xaxes(showticklabels=False)

            fig.update_layout(
                title_text="Distribution of network metrics for all datasets",
                height=1200,
                width=1200,
                margin=dict(l=4, r=4, t=4, b=4)  # Adjust the margins here
            )

            fig.write_image(
                str(folder / f"{filename}.pdf"), width=1500, height=1500, scale=5
            )
            # also to html
            fig.write_html(str(folder / f"{filename}.html"))

        plot_distribution_metrics(
            concerned_columns, df, summary_metrics_distribution_folder, "metrics"
        )

        # plot the distribution for each column with plotly in same figure, do it with scatter plot
        # transpose the x and y axises
        # get the figure larger, output in png

        # after the observation, we will need to do
        # num_edges => log scale
        # num_features => log scale
        # num_classes => log scale
        # k2 => log scale ?
        # k_max => log scale
        # do this first and check the distribution again

        df["ln_num_edges"] = np.log(df["num_edges"])
        # df["ln_num_features"] = np.log(df["num_features"])
        # find out the one with 0 for num_features
        df["ln_num_features"] = df.apply(
            lambda row: np.log(row["num_features"]) if row["num_features"] > 0 else 0,
            axis=1,
        )
        df["ln_num_classes"] = np.log(df["num_classes"])
        df["ln_k2"] = np.log(df["k2"])
        df["ln_k_max"] = np.log(df["k_max"])
        df["ln_d_max_random"] = np.log(df["d_max_random"])
        df["ln_average_shortest_path_length"] = np.log(
            df["average_shortest_path_length"]
        )
        df["ln_diameter"] = np.log(df["diameter"])

        ln_concerned_columns = [
            "ln_num_edges",
            "ln_num_features",
            "ln_num_classes",
            "ln_n",
            "k",
            "std_k",
            "ln_k2",
            "k_min",
            "ln_k_max",
            "ln_d_max_random",
            "gamma_degree",
            "ln_average_shortest_path_length",
            "ln_diameter",
            "average_clustering_coefficient",
            "transitivity",
            # p_k_max
            # ks_statistic
        ]
        plot_distribution_metrics(
            ln_concerned_columns,
            df,
            summary_metrics_distribution_folder,
            "metrics_ln",
        )
        # and then do min-max normalization for each column ln_concerned_columns
        # and then plot the distribution again
        # do the min-max normalization
        for column in ln_concerned_columns:
            df[column] = (df[column] - df[column].min()) / (
                    df[column].max() - df[column].min()
            )
        plot_distribution_metrics(
            ln_concerned_columns,
            df,
            summary_metrics_distribution_folder,
            "metrics_ln_min_max",
        )
        df.to_csv(summary_folder / "summary_network_metrics_min_max.csv", index=False)

    @staticmethod
    def plot_selected_metrics():
        summary_folder = REPORT_DIR / "summary"
        summary_csv_scaled = summary_folder / "summary_network_metrics_min_max.csv"
        summary_metrics_distribution_folder = summary_folder / "metrics_distribution"
        # plot tsne with these row combinations
        row_columns_combinations = [
            ("ln_num_edges", "ln_num_classes", "ln_average_shortest_path_length"),
            ("ln_num_classes", "ln_n", "ln_k_max"),
            ("ln_num_classes", "ln_n", "ln_average_shortest_path_length"),
            (
                "ln_num_classes",
                "ln_k_max",
                "ln_average_shortest_path_length",
            ),
            (
                "ln_num_classes",
                "ln_average_shortest_path_length",
                "ln_diameter",
            ),
        ]

        for row_columns_combination in row_columns_combinations:
            df = pd.read_csv(summary_csv_scaled, index_col=False)
            # drop na for the average shortest path length and diameter
            df = df.dropna(subset=["average_shortest_path_length", "diameter"])
            # dataset is in the finished dataset
            df = df[df["dataset"].isin(finished_dataset)]
            logger.info(f"shape: {df.shape}")

            # only include the columns in the row_columns_combination
            features = df[list(row_columns_combination) + ["dataset"]]
            # plot the 3D scatter plot
            fig = px.scatter_3d(
                features,
                x=row_columns_combination[0],
                y=row_columns_combination[1],
                z=row_columns_combination[2],
                color="dataset",
            )
            # fig.show()
            fig.write_html(
                str(
                    summary_metrics_distribution_folder
                    / f"3d_metric_{'_'.join(row_columns_combination)}.html"
                )
            )
            fig.write_image(
                str(
                    summary_metrics_distribution_folder
                    / f"3d_metric_{'_'.join(row_columns_combination)}.pdf"
                ),
                width=1200,
                height=600,
                scale=5,
            )

    @staticmethod
    def fix_k():
        # read the summary csv again
        summary_folder = REPORT_DIR / "summary"
        summary_csv = summary_folder / "summary_network_metrics.csv"
        if not summary_csv.exists():
            raise ValueError(f"Summary csv {summary_csv} does not exist")

        df = pd.read_csv(summary_csv, index_col=False)
        # recalculate the k by 2*E/N
        df["k"] = df.apply(lambda row: 2 * row["num_edges"] / row["num_nodes"], axis=1)
        df.to_csv(
            summary_csv, index=False
        )

    @staticmethod
    def classify_need_graph():

        dataset_classification_mapping = {
            "Actor": 'cross',
            "AMAZON_COMPUTERS": 'cross',
            "AMAZON_PHOTO": 'cross',
            "AttributedGraphDataset_BlogCatalog": 'cross',
            "AttributedGraphDataset_CiteSeer": 'circle',
            "AttributedGraphDataset_Cora": 'circle',
            "AttributedGraphDataset_Flickr": 'cross',
            "AttributedGraphDataset_Pubmed": 'cross',
            "AttributedGraphDataset_Wiki": 'cross',
            "CitationsFull_CiteSeer": 'circle',
            "CitationsFull_Cora": 'circle',
            "CitationsFull_Cora_ML": 'circle',
            "CitationsFull_DBLP": 'circle',
            "CitationsFull_PubMed": 'cross',
            "CiteSeer": 'circle',
            "Coauther_CS": 'circle',
            "Coauther_Physics": 'circle',
            "Cora": 'circle',
            "Cora_Full": 'circle',
            "GitHub": 'cross',
            "HeterophilousGraphDataset_Amazon_ratings": 'cross',
            "HeterophilousGraphDataset_Minesweeper": 'circle',
            "HeterophilousGraphDataset_Questions": 'circle',
            "HeterophilousGraphDataset_Roman_empire": 'circle',
            "HeterophilousGraphDataset_Tolokers": 'circle',
            "PubMed": 'cross',
            "TWITCH_DE": 'circle',
            "TWITCH_EN": 'circle',
            "TWITCH_ES": 'circle',
            "TWITCH_FR": 'circle',
            "TWITCH_PT": 'circle',
            "TWITCH_RU": 'circle',
            "WEBKB_Cornell": 'circle',
            "WEBKB_Texas": 'circle',
            "WEBKB_Wisconsin": 'circle',
        }
        dataset_benefical_dict = {'AttributedGraphDataset_Flickr': 'cross',
                                  'AttributedGraphDataset_BlogCatalog': 'cross', 'WEBKB_Wisconsin': 'cross',
                                  'Actor': 'cross', 'WEBKB_Texas': 'cross',
                                  'AttributedGraphDataset_Wiki': 'cross',
                                  'HeterophilousGraphDataset_Amazon_ratings': 'cross', 'PubMed': 'cross',
                                  'CitationsFull_PubMed': 'cross', 'CiteSeer': 'cross',
                                  'WEBKB_Cornell': 'cross', 'AMAZON_PHOTO': 'cross',
                                  'AttributedGraphDataset_Pubmed': 'cross', 'AMAZON_COMPUTERS': 'cross',
                                  'TWITCH_EN': 'circle', 'GitHub': 'circle', 'TWITCH_DE': 'circle',
                                  'Coauther_CS': 'circle', 'Coauther_Physics': 'circle',
                                  'AttributedGraphDataset_CiteSeer': 'circle',
                                  'HeterophilousGraphDataset_Tolokers': 'circle', 'TWITCH_PT': 'circle',
                                  'Cora': 'circle', 'TWITCH_FR': 'circle', 'CitationsFull_CiteSeer': 'circle',
                                  'TWITCH_ES': 'circle', 'HeterophilousGraphDataset_Roman_empire': 'circle',
                                  'AttributedGraphDataset_Cora': 'circle', 'CitationsFull_DBLP': 'circle',
                                  'TWITCH_RU': 'circle', 'HeterophilousGraphDataset_Questions': 'circle',
                                  'HeterophilousGraphDataset_Minesweeper': 'circle', 'Cora_Full': 'circle',
                                  'CitationsFull_Cora_ML': 'circle', 'CitationsFull_Cora': 'circle'}

        summary_folder = REPORT_DIR / "summary"
        summary_csv_scaled = summary_folder / "summary_network_metrics_min_max.csv"

        row_columns_combination = ["num_edges", "num_classes", "average_shortest_path_length"]
        df = pd.read_csv(summary_csv_scaled, index_col=False)
        # drop na for the average shortest path length and diameter
        logger.info(f"columns: {df.columns}")
        df = df.dropna(subset=["average_shortest_path_length", "diameter"])
        # dataset is in the finished dataset
        df = df[df["dataset"].isin(finished_dataset)]
        logger.info(f"shape: {df.shape}")
        logger.info(list(row_columns_combination) + ["dataset"])
        df_features = df[list(row_columns_combination) + ["dataset"]].copy(deep=True)
        df_features['symbol'] = df_features['dataset'].map(dataset_benefical_dict)
        logger.info(df_features.columns.tolist())
        # drop dataset column, and then do the classification with the first three columns, target is the symbol
        # df_features = df_features.drop(columns=['dataset'])
        logger.info(df_features.columns.tolist())
        # do the classification
        X = df_features.drop(columns=['symbol', "dataset"])
        y = df_features['symbol']
        logger.info(f"X: {X.columns.tolist()}")
        logger.debug(f"y: {y}")

        # train a decision tree
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=0, max_depth=3)
        clf.fit(X, y)
        logger.info(f"feature importance: {clf.feature_importances_}")
        logger.info(f"feature importance: {X.columns.tolist()}")
        logger.info(f"feature importance: {clf.feature_importances_.tolist()}")
        # do the prediction
        y_pred = clf.predict(X)
        logger.info(f"y_pred: {len(y_pred)}")
        # show the accuracy
        from sklearn.metrics import accuracy_score
        logger.info(f"accuracy: {accuracy_score(y, y_pred)}")
        # show the confusion matrix
        from sklearn.metrics import confusion_matrix
        logger.info(f"confusion matrix: {confusion_matrix(y, y_pred)}")
        # plot the decision tree
        from sklearn import tree
        fig = plt.figure(figsize=(25, 20))
        _ = tree.plot_tree(clf,
                           feature_names=X.columns.tolist(),
                           class_names=['cross', 'circle'],
                           filled=True)
        # add a text, "cross is not beneficial, circle is beneficial"
        plt.text(0.5, 0.5, "cross is not beneficial, circle is beneficial", fontsize=14,
                 transform=plt.gcf().transFigure)
        # save pdf and png
        fig.savefig(REPORT_DIR / "summary" / "network_metrics_decision_tree.pdf")
        fig.savefig(REPORT_DIR / "summary" / "network_metrics_decision_tree.png")
        logger.info(f"wrong prediction: {df_features[y != y_pred][['dataset', 'symbol']]}")

    @staticmethod
    def classify_node2vec():
        node2vec_dict = {'TWITCH_FR': 'cross', 'Actor': 'cross', 'HeterophilousGraphDataset_Roman_empire': 'cross',
                         'WEBKB_Wisconsin': 'cross', 'HeterophilousGraphDataset_Minesweeper': 'cross',
                         'AttributedGraphDataset_Pubmed': 'cross', 'AttributedGraphDataset_Flickr': 'cross',
                         'AttributedGraphDataset_CiteSeer': 'cross', 'AttributedGraphDataset_Cora': 'cross',
                         'WEBKB_Texas': 'cross', 'HeterophilousGraphDataset_Amazon_ratings': 'cross',
                         'TWITCH_RU': 'cross', 'WEBKB_Cornell': 'cross', 'AttributedGraphDataset_BlogCatalog': 'cross',
                         'CiteSeer': 'cross', 'HeterophilousGraphDataset_Questions': 'cross',
                         'AttributedGraphDataset_Wiki': 'cross', 'CitationsFull_CiteSeer': 'cross',
                         'TWITCH_EN': 'circle', 'Cora': 'circle', 'TWITCH_DE': 'circle', 'TWITCH_PT': 'circle',
                         'TWITCH_ES': 'circle', 'PubMed': 'circle', 'CitationsFull_PubMed': 'circle',
                         'Coauther_CS': 'circle', 'HeterophilousGraphDataset_Tolokers': 'circle',
                         'Coauther_Physics': 'circle', 'GitHub': 'circle', 'AMAZON_PHOTO': 'circle',
                         'AMAZON_COMPUTERS': 'circle', 'CitationsFull_Cora': 'circle',
                         'CitationsFull_Cora_ML': 'circle', 'Cora_Full': 'circle', 'CitationsFull_DBLP': 'circle'}

        summary_folder = REPORT_DIR / "summary"
        summary_csv_scaled = summary_folder / "summary_network_metrics_min_max.csv"

        row_columns_combination = ["num_edges", "num_classes", "average_shortest_path_length"]
        df = pd.read_csv(summary_csv_scaled, index_col=False)
        # drop na for the average shortest path length and diameter
        logger.info(f"columns: {df.columns}")
        df = df.dropna(subset=["average_shortest_path_length", "diameter"])
        # dataset is in the finished dataset
        df = df[df["dataset"].isin(finished_dataset)]
        logger.info(f"shape: {df.shape}")
        logger.info(list(row_columns_combination) + ["dataset"])
        df_features = df[list(row_columns_combination) + ["dataset"]].copy(deep=True)
        df_features['symbol'] = df_features['dataset'].map(node2vec_dict)
        logger.info(df_features.columns.tolist())
        # drop dataset column, and then do the classification with the first three columns, target is the symbol
        # df_features = df_features.drop(columns=['dataset'])
        logger.info(df_features.columns.tolist())
        # do the classification
        X = df_features.drop(columns=['symbol', "dataset"])
        y = df_features['symbol']
        logger.info(f"X: {X.columns.tolist()}")
        logger.debug(f"y: {y}")

        # train a decision tree
        clf = DecisionTreeClassifier(random_state=0, max_depth=3)
        clf.fit(X, y)
        logger.info(f"feature importance: {clf.feature_importances_}")
        logger.info(f"feature importance: {X.columns.tolist()}")
        logger.info(f"feature importance: {clf.feature_importances_.tolist()}")
        # do the prediction
        y_pred = clf.predict(X)
        logger.info(f"y_pred: {len(y_pred)}")
        # show the accuracy
        from sklearn.metrics import accuracy_score
        logger.info(f"accuracy: {accuracy_score(y, y_pred)}")
        # show the confusion matrix
        from sklearn.metrics import confusion_matrix
        logger.info(f"confusion matrix: {confusion_matrix(y, y_pred)}")
        # plot the decision tree
        from sklearn import tree
        fig = plt.figure(figsize=(25, 20))
        _ = tree.plot_tree(clf,
                           feature_names=X.columns.tolist(),
                           class_names=['cross', 'circle'],
                           filled=True)
        # add a text, "cross is not beneficial, circle is beneficial"
        plt.text(0.5, 0.5, "cross is node2vec not , circle is beneficial", fontsize=14,
                 transform=plt.gcf().transFigure)
        # save pdf and png
        fig.savefig(REPORT_DIR / "summary" / "network_metrics_decision_tree_node2vec.pdf")
        fig.savefig(REPORT_DIR / "summary" / "network_metrics_decision_tree_node2vec.png")
        logger.info(f"wrong prediction: {df_features[y != y_pred][['dataset', 'symbol']]}")

    @staticmethod
    def check_list_include_relations():
        node2vec_dict = {'TWITCH_FR': 'cross', 'Actor': 'cross', 'HeterophilousGraphDataset_Roman_empire': 'cross',
                         'WEBKB_Wisconsin': 'cross', 'HeterophilousGraphDataset_Minesweeper': 'cross',
                         'AttributedGraphDataset_Pubmed': 'cross', 'AttributedGraphDataset_Flickr': 'cross',
                         'AttributedGraphDataset_CiteSeer': 'cross', 'AttributedGraphDataset_Cora': 'cross',
                         'WEBKB_Texas': 'cross', 'HeterophilousGraphDataset_Amazon_ratings': 'cross',
                         'TWITCH_RU': 'cross', 'WEBKB_Cornell': 'cross', 'AttributedGraphDataset_BlogCatalog': 'cross',
                         'CiteSeer': 'cross', 'HeterophilousGraphDataset_Questions': 'cross',
                         'AttributedGraphDataset_Wiki': 'cross', 'CitationsFull_CiteSeer': 'cross',
                         'TWITCH_EN': 'circle', 'Cora': 'circle', 'TWITCH_DE': 'circle', 'TWITCH_PT': 'circle',
                         'TWITCH_ES': 'circle', 'PubMed': 'circle', 'CitationsFull_PubMed': 'circle',
                         'Coauther_CS': 'circle', 'HeterophilousGraphDataset_Tolokers': 'circle',
                         'Coauther_Physics': 'circle', 'GitHub': 'circle', 'AMAZON_PHOTO': 'circle',
                         'AMAZON_COMPUTERS': 'circle', 'CitationsFull_Cora': 'circle',
                         'CitationsFull_Cora_ML': 'circle', 'Cora_Full': 'circle', 'CitationsFull_DBLP': 'circle'}

        need_graph_dict = {'AttributedGraphDataset_Flickr': 'cross',
                           'AttributedGraphDataset_BlogCatalog': 'cross', 'WEBKB_Wisconsin': 'cross',
                           'Actor': 'cross', 'WEBKB_Texas': 'cross',
                           'AttributedGraphDataset_Wiki': 'cross',
                           'HeterophilousGraphDataset_Amazon_ratings': 'cross', 'PubMed': 'cross',
                           'CitationsFull_PubMed': 'cross', 'CiteSeer': 'cross',
                           'WEBKB_Cornell': 'cross', 'AMAZON_PHOTO': 'cross',
                           'AttributedGraphDataset_Pubmed': 'cross', 'AMAZON_COMPUTERS': 'cross',
                           'TWITCH_EN': 'circle', 'GitHub': 'circle', 'TWITCH_DE': 'circle',
                           'Coauther_CS': 'circle', 'Coauther_Physics': 'circle',
                           'AttributedGraphDataset_CiteSeer': 'circle',
                           'HeterophilousGraphDataset_Tolokers': 'circle', 'TWITCH_PT': 'circle',
                           'Cora': 'circle', 'TWITCH_FR': 'circle', 'CitationsFull_CiteSeer': 'circle',
                           'TWITCH_ES': 'circle', 'HeterophilousGraphDataset_Roman_empire': 'circle',
                           'AttributedGraphDataset_Cora': 'circle', 'CitationsFull_DBLP': 'circle',
                           'TWITCH_RU': 'circle', 'HeterophilousGraphDataset_Questions': 'circle',
                           'HeterophilousGraphDataset_Minesweeper': 'circle', 'Cora_Full': 'circle',
                           'CitationsFull_Cora_ML': 'circle', 'CitationsFull_Cora': 'circle'}
        need_graph_cross = []
        need_graph_circle = []
        for dataset, symbol in need_graph_dict.items():
            if symbol == 'cross':
                need_graph_cross.append(dataset)
            else:
                need_graph_circle.append(dataset)
        node2vec_cross = []
        node2vec_circle = []
        for dataset, symbol in node2vec_dict.items():
            if symbol == 'cross':
                node2vec_cross.append(dataset)
            else:
                node2vec_circle.append(dataset)
        logger.info(f"need graph cross: {need_graph_cross}")
        logger.info(f"need graph circle: {need_graph_circle}")
        logger.info(f"node2vec cross: {node2vec_cross}")
        logger.info(f"node2vec circle: {node2vec_circle}")
        # check if the node2vec cross is in the need graph cross
        logger.info(f"node2vec cross in need graph cross: {set(node2vec_cross).issubset(set(need_graph_cross))}")
        # get the intersection
        logger.info(f"node2vec cross and need graph cross: {set(node2vec_cross).intersection(set(need_graph_cross))}")
        # get the rest which is not in intersection
        logger.info(
            f"node2vec cross and need graph cross differences: {set(node2vec_cross).difference(set(need_graph_cross))}")

    @staticmethod
    def classify_q1andq2():
        need_graph_dict = {'AttributedGraphDataset_Flickr': 'cross',
                           'AttributedGraphDataset_BlogCatalog': 'cross', 'WEBKB_Wisconsin': 'cross',
                           'Actor': 'cross', 'WEBKB_Texas': 'cross',
                           'AttributedGraphDataset_Wiki': 'cross',
                           'HeterophilousGraphDataset_Amazon_ratings': 'cross', 'PubMed': 'cross',
                           'CitationsFull_PubMed': 'cross', 'CiteSeer': 'cross',
                           'WEBKB_Cornell': 'cross', 'AMAZON_PHOTO': 'cross',
                           'AttributedGraphDataset_Pubmed': 'cross', 'AMAZON_COMPUTERS': 'cross',
                           'TWITCH_EN': 'circle', 'GitHub': 'circle', 'TWITCH_DE': 'circle',
                           'Coauther_CS': 'circle', 'Coauther_Physics': 'circle',
                           'AttributedGraphDataset_CiteSeer': 'circle',
                           'HeterophilousGraphDataset_Tolokers': 'circle', 'TWITCH_PT': 'circle',
                           'Cora': 'circle', 'TWITCH_FR': 'circle', 'CitationsFull_CiteSeer': 'circle',
                           'TWITCH_ES': 'circle', 'HeterophilousGraphDataset_Roman_empire': 'circle',
                           'AttributedGraphDataset_Cora': 'circle', 'CitationsFull_DBLP': 'circle',
                           'TWITCH_RU': 'circle', 'HeterophilousGraphDataset_Questions': 'circle',
                           'HeterophilousGraphDataset_Minesweeper': 'circle', 'Cora_Full': 'circle',
                           'CitationsFull_Cora_ML': 'circle', 'CitationsFull_Cora': 'circle'}
        node2vec_dict = {'TWITCH_FR': 'cross', 'Actor': 'cross', 'HeterophilousGraphDataset_Roman_empire': 'cross',
                         'WEBKB_Wisconsin': 'cross', 'HeterophilousGraphDataset_Minesweeper': 'cross',
                         'AttributedGraphDataset_Pubmed': 'cross', 'AttributedGraphDataset_Flickr': 'cross',
                         'AttributedGraphDataset_CiteSeer': 'cross', 'AttributedGraphDataset_Cora': 'cross',
                         'WEBKB_Texas': 'cross', 'HeterophilousGraphDataset_Amazon_ratings': 'cross',
                         'TWITCH_RU': 'cross', 'WEBKB_Cornell': 'cross', 'AttributedGraphDataset_BlogCatalog': 'cross',
                         'CiteSeer': 'cross', 'HeterophilousGraphDataset_Questions': 'cross',
                         'AttributedGraphDataset_Wiki': 'cross', 'CitationsFull_CiteSeer': 'cross',
                         'TWITCH_EN': 'circle', 'Cora': 'circle', 'TWITCH_DE': 'circle', 'TWITCH_PT': 'circle',
                         'TWITCH_ES': 'circle', 'PubMed': 'circle', 'CitationsFull_PubMed': 'circle',
                         'Coauther_CS': 'circle', 'HeterophilousGraphDataset_Tolokers': 'circle',
                         'Coauther_Physics': 'circle', 'GitHub': 'circle', 'AMAZON_PHOTO': 'circle',
                         'AMAZON_COMPUTERS': 'circle', 'CitationsFull_Cora': 'circle',
                         'CitationsFull_Cora_ML': 'circle', 'Cora_Full': 'circle', 'CitationsFull_DBLP': 'circle'}
        # Load your datasets here
        # Replace these with the actual dataframes containing the features and 'symbol' columns for each classification
        # For instance, df_features_q1 and df_features_q2 could be your datasets for question 1 and question 2
        # df_features_q1 = pd.read_csv('your_dataset_q1.csv')
        # df_features_q2 = pd.read_csv('your_dataset_q2.csv')
        summary_folder = REPORT_DIR / "summary"
        summary_csv_scaled = summary_folder / "summary_network_metrics_min_max.csv"

        row_columns_combination = ["num_edges", "num_classes", "average_shortest_path_length"]
        df = pd.read_csv(summary_csv_scaled, index_col=False)
        logger.info(f"columns: {df.columns}")
        df = df.dropna(subset=["average_shortest_path_length", "diameter"])
        # dataset is in the finished dataset
        df = df[df["dataset"].isin(finished_dataset)]
        logger.info(f"shape: {df.shape}")
        logger.info(list(row_columns_combination) + ["dataset"])
        df_features_q1 = df[list(row_columns_combination) + ["dataset"]].copy(deep=True)
        df_features_q2 = df[list(row_columns_combination) + ["dataset"]].copy(deep=True)

        df_features_q1['symbol'] = df_features_q1['dataset'].map(need_graph_dict)
        df_features_q2['symbol'] = df_features_q2['dataset'].map(node2vec_dict)
        # Prepare the data for Q1
        X_q1 = df_features_q1.drop(columns=['symbol', 'dataset'])
        y_q1 = df_features_q1['symbol']
        clf_q1 = DecisionTreeClassifier(random_state=0, max_depth=3)
        clf_q1.fit(X_q1, y_q1)

        # Prepare the data for Q2
        X_q2 = df_features_q2.drop(columns=['symbol', 'dataset'])
        y_q2 = df_features_q2['symbol']
        clf_q2 = DecisionTreeClassifier(random_state=0, max_depth=3)
        clf_q2.fit(X_q2, y_q2)

        # Plot both trees
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(80, 20), dpi=150)

        # Plot for Q1
        tree_plot_q1 = plot_tree(clf_q1, feature_names=X_q1.columns.tolist(), class_names=['Yes', 'No'],
                                 filled=True, ax=axes[0], fontsize=32)

        axes[0].set_title('Q1: Benefit from graph representation learning?', fontsize=40, fontweight='bold')

        # Plot for Q2
        tree_plot_q2 = plot_tree(clf_q2, feature_names=X_q2.columns.tolist(), class_names=['Yes', 'No'],
                                 filled=True, ax=axes[1], fontsize=32)
        axes[1].set_title('Q2: Is structural information alone sufficient?', fontsize=40, fontweight='bold')

        plt.tight_layout()
        plt.savefig(summary_folder / 'decision_trees_q1_q2.pdf')
        plt.show()


if __name__ == "__main__":
    logger = get_logger("network_metrics")

    existing_datasets, summary_df = DatasetNetworkMetrics.exist_datasets()
    logger.info(f"Existing datasets: {existing_datasets}")
    success_datasets = []

    for dataset_enum in HOMO_DATASETS:
        if type(dataset_enum) is DataSetEnum:
            dataset_name = dataset_enum.value
        else:
            dataset_name = dataset_enum
        try:
            logger.info(f"Calculating network metrics for {dataset_name}")
            logger.info(f"Dataset existing: {dataset_name in existing_datasets}")
            to_be_calculated = DatasetNetworkMetrics.require_calculations(
                name=dataset_name, df=summary_df
            )
            if not len(to_be_calculated):
                continue
            with timer(logger, f"load dataset: {dataset_name}"):
                dataset = load_dataset(dataset_name)
            with timer(logger, f"to networkx: {dataset_name}"):
                data_graph = to_networkx(
                    dataset.data, to_undirected=not dataset.data.is_directed()
                )
            dataset_network_metric = DatasetNetworkMetrics(dataset_name, data_graph)

            if "graph_attribute" in to_be_calculated:
                with timer(logger, "Calculate graph attributes"):
                    dataset_network_metric.calculate_graph_attribute()
            if "calculate_second_moment" in to_be_calculated:
                with timer(logger, "Calculate second moment"):
                    dataset_network_metric.calculate_second_moment()
            if "calculate_degree_exponent" in to_be_calculated:
                with timer(logger, "Calculate degree exponent"):
                    dataset_network_metric.calculate_degree_exponent()
            if "calculate_gamma_regime" in to_be_calculated:
                with timer(logger, "Calculate gamma regime"):
                    dataset_network_metric.calculate_gamma_regime()
            if "calculate_k_max" in to_be_calculated:
                with timer(logger, "Calculate k_max"):
                    dataset_network_metric.calculate_p_k_max()
            if "calculate_average_shortest_path_length" in to_be_calculated:
                try:
                    with timer(logger, "Calculate average shortest path length"):
                        dataset_network_metric.calculate_average_shortest_path_length()
                except Exception as e_path:
                    logger.error(str(e_path))
            if "calculate_diameter" in to_be_calculated:
                with timer(logger, "Calculate diameter"):
                    dataset_network_metric.calculate_diameter()
            if "calculate_num_classes" in to_be_calculated:
                with timer(logger, "Calculate num classes"):
                    dataset_network_metric.calculate_num_classes(dataset)

            if "calculate_num_features" in to_be_calculated:
                with timer(logger, "Calculate num features"):
                    dataset_network_metric.calculate_num_features(dataset)
            success_datasets.append(dataset_name)
        except Exception as e:
            logger.error(f"Error while calculating network metrics for {dataset_name}")
            logger.exception(e)

    logger.info(f"Successfully calculated network metrics for {success_datasets}")

    # fix the k, recalculate it by 2*E/N
    DatasetNetworkMetrics.fix_k()

    # plot the k and ln n
    DatasetNetworkMetrics.plot_summary_k_and_ln_n_random_network()

    # plot the k and standard deviation against random network model
    DatasetNetworkMetrics.plot_summary_k_and_std_random_network()

    # plot tsne for the summary_network_metrics.csv
    DatasetNetworkMetrics.plot_summary_tsne()

    # explore each metric distribution for all dataset, and then do min-max normalization
    DatasetNetworkMetrics.explore_metric_distribution()

    # plot the selected metrics for each dataset
    DatasetNetworkMetrics.plot_selected_metrics()
    DatasetNetworkMetrics.classify_need_graph()
    DatasetNetworkMetrics.classify_node2vec()
    DatasetNetworkMetrics.check_list_include_relations()
    DatasetNetworkMetrics.classify_q1andq2()
