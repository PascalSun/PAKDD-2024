from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from sklearn.manifold import TSNE

from src.utils.logger import get_logger
from src.utils.timer import timer

logger = get_logger()


def plot_tsne(
    emb_df,
    label_field_name: str = "y",
    title: str = "TSNE Plot",
    filename: Optional[Path] = None,
    label_int_2_str: Optional[dict] = None,
    show_in_web: bool = False,
):
    """
    :param emb_df:
    :param label_field_name:
    :param title:
    :param filename:
    :param label_int_2_str: a dictionary mapping label int to label str
    :param show_in_web: if True, show the plot in web browser
    :return:
    """
    emb_df_no_labels = emb_df.drop(
        columns=[label_field_name]
    )  # remove the labels column
    tsne = TSNE(n_components=2, random_state=42)
    emb_df_2d = tsne.fit_transform(emb_df_no_labels)

    # Create the scatter plot with colors according to labels
    unique_labels = emb_df[label_field_name].unique()
    fig = go.Figure()
    for label in unique_labels:
        mask = emb_df[label_field_name] == label
        logger.info(f"Plotting {label} with {mask.sum()} samples")
        fig.add_trace(
            go.Scatter(
                x=emb_df_2d[mask][:, 0],
                y=emb_df_2d[mask][:, 1],
                mode="markers",
                name=label_int_2_str[label]
                if label_int_2_str is not None
                else str(label),
            )
        )
    fig.update_layout(
        title=title,
    )
    # Show the plot
    if show_in_web:
        fig.show()
    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        with timer(logger, "Saving TSNE Plot"):
            # save the figure
            fig.write_image(filename)
