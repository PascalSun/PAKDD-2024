from pathlib import Path
from typing import Optional

from gqlalchemy import Memgraph
from neo4j import GraphDatabase
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected

from src.iid.dataset.load_datasets import load_dataset
from src.iid.graph_metrics.cypher_queries import (
    AVERAGE_IN_OUT_DEGREE_QUERY,
    IN_OUT_DEGREE_QUERY,
    SUPER_GRAPH_DEGREE,
    SUPER_GRAPH_QUERY,
)
from src.iid.graph_metrics.models import GraphEngine
from src.iid.graph_metrics.parser import parse_args
from src.utils.constants import REPORT_DIR
from src.utils.logger import get_logger
from src.utils.timer import timer


class DatasetToCypher:
    """
    1. Load the dataset, e.g. Cora, we will need to give it text labels, so the graph will be more readable
    2. Run superGraph to get the supergraph
    3. Extract the metrics for the graph

    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        user: str = "neo4j",
        password: str = "verystrongpassword",
        clean_db: bool = True,
        graph_engine: str = GraphEngine.MEMGRAPH.value,
    ):
        self.neo4j_driver = None
        self.mem_graph_driver = None
        if graph_engine == GraphEngine.NEO4J.value:
            self.neo4j_driver = GraphDatabase.driver(
                f"bolt://{host}:{port}", auth=(user, password)
            )
            # clean the database
            if clean_db:
                with self.neo4j_driver.session() as session:
                    session.run("MATCH (a) DETACH DELETE a")
        elif graph_engine == GraphEngine.MEMGRAPH.value:
            self.mem_graph_driver = Memgraph(
                host=host, port=port, username=user, password=password
            )
            if clean_db:
                self.mem_graph_driver.execute("MATCH (a) DETACH DELETE a")
        else:
            raise ValueError(f"Graph engine {graph_engine} is not supported")
        self.graph_engine = graph_engine
        self.logger = get_logger()

    def execute_load_graph_query(self, data: Data):
        """
        Load the graph to Neo4j/Memgraph
        Parameters
        ----------
        data

        Returns
        -------

        """
        self.logger.info(f"Is the graph undirected: {is_undirected(data.edge_index)}")
        if self.graph_engine == GraphEngine.NEO4J.value:
            with self.neo4j_driver.session() as session:
                for query in self.to_load_cypher(data):
                    session.run(query)
        elif self.graph_engine == "memgraph":
            for query in self.to_load_cypher(data):
                self.mem_graph_driver.execute(query)
        else:
            raise ValueError(f"Graph engine {self.graph_engine} is not supported")

    def execute_super_graph(self):
        if self.graph_engine == GraphEngine.NEO4J.value:
            self.neo4j_driver.session().run(SUPER_GRAPH_QUERY)
        elif self.graph_engine == GraphEngine.MEMGRAPH.value:
            self.mem_graph_driver.execute(SUPER_GRAPH_QUERY)
        else:
            raise ValueError(f"Graph engine {self.graph_engine} is not supported")

    def execute_supergraph_metrics(self, graph_metrics_dir: Optional[Path] = None):
        """
        Output it them into the csv files
        The file will be saved to the folder graph_metrics
        Returns
        -------

        """
        if self.graph_engine == GraphEngine.NEO4J.value:
            with self.neo4j_driver.session() as session:
                super_node_results = session.run(SUPER_GRAPH_DEGREE)
                in_out_degree_results = session.run(IN_OUT_DEGREE_QUERY)
                average_in_out_degree_results = session.run(AVERAGE_IN_OUT_DEGREE_QUERY)

        elif self.graph_engine == GraphEngine.MEMGRAPH.value:
            super_node_results = self.mem_graph_driver.execute_and_fetch(
                SUPER_GRAPH_DEGREE
            )
            in_out_degree_results = self.mem_graph_driver.execute_and_fetch(
                IN_OUT_DEGREE_QUERY
            )
            average_in_out_degree_results = self.mem_graph_driver.execute_and_fetch(
                AVERAGE_IN_OUT_DEGREE_QUERY
            )
        else:
            raise ValueError(f"Graph engine {self.graph_engine} is not supported")

        # save the results to the csv file
        if not graph_metrics_dir:
            graph_metrics_dir = REPORT_DIR / "graph_metrics"
        graph_metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(graph_metrics_dir / "super_graph_degree.csv", "w") as f:
            f.write("label,in_degree,out_degree\n")
            for result in super_node_results:
                f.write(
                    f"{result['label']},{result['inDegree']},{result['outDegree']}\n"
                )
        with open(graph_metrics_dir / "in_out_degree.csv", "w") as f:
            f.write("node_id,in_degree,out_degree,label\n")
            for result in in_out_degree_results:
                f.write(
                    f"{result['n.node_id']},{result['in_degree']},{result['out_degree']},{result['class_label']}\n"
                )
        with open(graph_metrics_dir / "average_in_out_degree.csv", "w") as f:
            f.write("label,avg_in_degree,avg_out_degree\n")
            for result in average_in_out_degree_results:
                f.write(
                    f"{result['class']},{result['avg_in_degree']},{result['avg_out_degree']}\n"
                )

    @staticmethod
    def to_load_cypher(data: Data, undirected: bool = True):
        """
        Depending on whether the graph is directed or undirected, we will need to create the relationship differently

        Parameters
        ----------
        data: Data
            the data object from pytorch geometric
        undirected: bool, default = True
            whether the graph is undirected
        """
        cypher_queries = []

        # create nodes
        for node_index, node_feature in enumerate(data.x):
            query = f"""
            CREATE (n:Node {{feature: {node_feature.tolist()},
                             node_id: {str(node_index)},
                             label: '{str(data.labels[data.y[node_index].item()])}'}})
            """
            cypher_queries.append(query)

        processed_edges = set()

        for edge_index in data.edge_index.t().tolist():
            edge_tuple = tuple(edge_index)
            if undirected:
                # Skip if the reverse edge was already processed
                if edge_tuple[::-1] in processed_edges:
                    continue
                processed_edges.add(edge_tuple)
            query = f"""
                    MATCH (n {{node_id: {edge_index[0]}}}), (m {{node_id: {edge_index[1]}}})
                    CREATE (n)-[:CONNECTED_TO]->(m)
                """
            cypher_queries.append(query)
        return cypher_queries


if __name__ == "__main__":
    # take parameters from the command line
    args = parse_args()
    # load dataset
    dataset_name = args.dataset
    dataset = load_dataset(dataset_name)
    dataset_to_cypher = DatasetToCypher(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        clean_db=args.clean_db,
        graph_engine=args.graph_engine,
    )
    dataset_to_cypher.logger.info(f"Loaded dataset {args}")
    if args.clean_db:
        with timer(
            dataset_to_cypher.logger,
            f"Load graph to {args.dataset}/{args.graph_engine}",
        ):
            dataset_to_cypher.execute_load_graph_query(dataset[0])

    with timer(dataset_to_cypher.logger, "Create supergraph"):
        dataset_to_cypher.execute_super_graph()

    with timer(dataset_to_cypher.logger, "Get supergraph metrics"):
        dataset_to_cypher.execute_supergraph_metrics(
            REPORT_DIR / dataset_name / "graph_metrics"
        )
