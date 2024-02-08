import logging
import time
from logging import Logger
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from torch_geometric.data import Data, InMemoryDataset

logger = logging.getLogger(__name__)

# fill here
DATA_DIR = "."


class timer:
    """
    util function used to log the time taken by a part of program
    """

    def __init__(self, logger: Logger, message: str):
        """
        init the timer

        Parameters
        ----------
        logger: Logger
            logger to write the logs
        message: str
            message to log, like start xxx
        """
        self.message = message
        self.logger = logger
        self.start = 0
        self.duration = 0
        self.sub_timers = []

    def __enter__(self):
        """
        context enter to start write this
        """
        self.start = time.time()
        self.logger.info("Starting %s" % self.message)
        return self

    def __exit__(self, context, value, traceback):
        """
        context exit will write this
        """
        self.duration = time.time() - self.start
        self.logger.info(f"Finished {self.message}, that took {self.duration:.3f}")


class SmartGrid(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        self.logger = logging.getLogger("SmartGrid")
        super(SmartGrid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        pass

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_file_names(self):
        return ["GSX.csv", "i9bust.csv"]

    @staticmethod
    def adj_to_edge_index(adj_matrix):
        """
        Return the edge_index and the graph
        Parameters
        ----------
        adj_matrix

        Returns
        -------

        """
        # Convert adjacency matrix to networkx graph
        graph = nx.from_numpy_array(adj_matrix)

        # Get the edge list from the graph
        edge_list = list(graph.edges())

        # Convert edge list to edge index
        edge_index = [[edge[0] for edge in edge_list], [edge[1] for edge in edge_list]]

        return torch.tensor(edge_index), graph

    def process(self):
        adj_matrix_df = pd.read_csv(
            Path(self.root) / "raw" / self.raw_file_names[0], header=None
        ).values.astype(float)
        original_data_df = pd.read_csv(
            Path(self.root) / "raw" / self.raw_file_names[1], header=None
        ).values.astype(float)

        # first process the graph
        self.logger.info(adj_matrix_df)
        edge_index, _ = self.adj_to_edge_index(adj_matrix_df)

        x_df = original_data_df[:, :9]
        y_df = original_data_df[:20000, 9:]

        x = torch.tensor(x_df)
        y = torch.tensor(y_df)
        self.logger.info(x.shape)

        # 定义新的维度
        num_groups = 20000
        rows_per_group = 6

        # 确保可以平均地将原始张量分成这些组
        assert x.size(0) == num_groups * rows_per_group

        # we will try to create 20000 graphs, which is 20000 Data objects here
        data_list = []

        for i in range(num_groups):
            # Extract a single chunk of x and y for the current graph
            x_chunk = x[i * rows_per_group : (i + 1) * rows_per_group]
            y_chunk = y[i : i + 1]

            # Reshape x_chunk to match the desired shape
            x_chunk = x_chunk.view(rows_per_group, -1)

            # Create a Data object for the current graph
            data_list.append(Data(x=x_chunk, edge_index=edge_index, y=y_chunk))

        torch.save(self.collate(data_list), self.processed_paths[0])


def plot_tsne(df, label_field_name: str = "y"):
    # Assuming your DataFrame is loaded into 'df'
    df.y = df.y.astype(int).astype(str)
    x_df = df.drop(columns=[label_field_name])
    y_df = df[[label_field_name]]

    # Perform t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=3, random_state=0)
    x_2d_tsne = tsne.fit_transform(x_df)
    unique_labels = y_df.y.unique()
    logger.info(f"Unique labels: {unique_labels}")
    # Create a scatter plot
    fig = go.Figure()
    for label in unique_labels:
        mask = df[label_field_name] == label
        logger.info(f"Plotting {label} with {mask.sum()} samples")
        fig.add_trace(
            go.Scatter(
                x=x_2d_tsne[mask][:, 0],
                y=x_2d_tsne[mask][:, 1],
                mode="markers",
                name=label,
            )
        )
    fig.update_layout(
        title="test",
    )
    fig.show()
    fig.write_image("tsne.png")


def plot_corr(df: pd.DataFrame):
    """
    Plot the correlation matrix for the dataframe
    Parameters
    ----------
    df

    Returns
    -------

    """
    # plot correlation matrix for x_df
    corr = df.corr()
    logger.info(corr)
    # Create annotations (correlation values)
    annotations = []
    for i, row in enumerate(corr.values):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": corr.columns[j],
                    "y": corr.index[i],
                    "font": {"color": "white" if abs(value) > 0.5 else "black"},
                    "text": str(round(value, 2)),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False,
                }
            )
    # plot with plotly, and put the values in the cells
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            hoverongaps=False,
            colorscale="RdBu_r",  # Diverging colorscale
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Correlation",
                titleside="top",
                tickmode="array",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "0.5", "1"],
                ticks="outside",
            ),
        )
    )

    # Add annotations
    fig.update_layout(annotations=annotations)

    # Adjust size if needed
    fig.update_layout(autosize=False, width=600, height=600)

    fig.show()
    fig.write_image("corr.png")


def plot_scatter(df: pd.DataFrame, save_path: str):
    features = df.columns
    n_features = len(features)

    # Create subplot grid
    fig = make_subplots(
        rows=n_features,
        cols=n_features,
        subplot_titles=tuple([f"{i} vs {j}" for i in features for j in features]),
    )

    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            logger.info(f"Plotting {feature_i} vs {feature_j}")
            row, col = i + 1, j + 1

            if i == j:
                # Diagonal: histogram
                fig.add_trace(
                    go.Histogram(x=df[feature_i], nbinsx=30), row=row, col=col
                )
            else:
                # Off-diagonal: scatter plot
                fig.add_trace(
                    go.Scatter(x=df[feature_i], y=df[feature_j], mode="markers"),
                    row=row,
                    col=col,
                )

    logger.info("Updating layout")
    # Update layout
    fig.update_layout(title_text="Scatter and Histogram Matrix", showlegend=False)
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    with timer(logger, "Saving figure"):
        # Save the figure
        fig.write_html(save_path)


if __name__ == "__main__":
    dataset = SmartGrid(root=DATA_DIR / "grid", name="smart_grid")

    # plot the tsne of the dataset for all of them
    # we first need to transfer the 20000 graphs into a single dataframe, with x and y
    data_csv = Path("data.csv")
    if data_csv.exists():
        data_df = pd.read_csv(data_csv)
    else:
        data_df = pd.DataFrame()
        for i in range(len(dataset)):
            # construct a df, for the graph, x will be 9 rows, 6 features, and y will be 9
            data = dataset[i]
            x = data.x.numpy()
            y = data.y.numpy()
            x_df = pd.DataFrame(x)
            # transpose the x_df to 9 rows, 6 features
            x_df = x_df.transpose()

            y_df = pd.DataFrame(y)
            y_df = y_df.transpose()
            df = pd.concat([x_df, y_df], axis=1)
            # add a column for the node id, which is the index
            df["node_id"] = df.index
            data_df = pd.concat([data_df, df], axis=0)
        data_df.columns = ["v1", "v2", "v3", "i1", "i2", "i3", "y", "node_id"]
        data_df.to_csv("data.csv", index=False)

    # get all node features and y for the node id
    node_df = data_df[data_df["node_id"] == 0].copy(deep=True)

    # drop the last column, which is the node id, split the data into x and y df
    x_df = node_df.drop(columns=["node_id"])
    y_df = node_df[["y"]]
    with timer(logger, "Plotting scatter"):
        plot_scatter(x_df, "scatter.html")
    with timer(logger, "Plotting corr"):
        plot_corr(x_df)
    with timer(logger, "Plotting tsne"):
        plot_tsne(x_df)
