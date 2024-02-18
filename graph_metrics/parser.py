import argparse
from typing import List, Optional

from graph_metrics.models import GraphEngine


def parse_args(arguments: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Load graph and calculate metrics")
    parser.add_argument(
        "-d", "--dataset", default="Cora", type=str, help="dataset name"
    )
    parser.add_argument(
        "-host", "--host", default="localhost", type=str, help="host name"
    )
    parser.add_argument("-port", "--port", default=7688, type=int, help="port number")
    parser.add_argument(
        "-user", "--user", default="memgraph", type=str, help="user name"
    )
    parser.add_argument(
        "-password",
        "--password",
        default="verystrongpassword",
        type=str,
        help="password",
    )

    parser.add_argument(
        "-clean_db",
        "--clean_db",
        dest="clean_db",
        action="store_true",
        help="clean the db",
    )
    parser.add_argument(
        "-no_clean_db",
        "--no_clean_db",
        dest="clean_db",
        action="store_false",
        help="do not clean the db",
    )
    parser.set_defaults(clean_db=True)

    parser.add_argument(
        "-graph_engine",
        "--graph_engine",
        default=GraphEngine.MEMGRAPH.value,
        type=str,
        choices=[GraphEngine.MEMGRAPH.value, GraphEngine.NEO4J.value],
        help="graph engine",
    )
    return parser.parse_args(arguments)
