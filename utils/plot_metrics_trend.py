from pathlib import Path
from typing import Optional

import plotly.graph_objects as go


def plot_metrics(
    metrics: dict,
    title: str,
    x_title: str,
    y_title: str,
    metric_name: str = "accuracy_score",
    filename: Optional[Path] = None,
) -> None:
    """
    This function is used to plot metrics trend for a given metric
    for example, dimension vs accuracy score
    :param metrics:
    :param title:
    :param x_title:
    :param y_title:
    :param metric_name:
    :param filename:
    :return:
    """
    metric_svm = []
    metric_knn = []
    metric_rf = []
    train_metric_svm = []
    train_metric_knn = []
    train_metric_rf = []
    for emb_dim in metrics.keys():
        metric_svm.append(
            [metric for metric in metrics[emb_dim] if metric["name"] == "SVM"][0][
                metric_name
            ]
        )
        metric_knn.append(
            [metric for metric in metrics[emb_dim] if metric["name"] == "KNN"][0][
                metric_name
            ]
        )
        metric_rf.append(
            [metric for metric in metrics[emb_dim] if metric["name"] == "RF"][0][
                metric_name
            ]
        )
        train_metric_svm.append(
            [metric for metric in metrics[emb_dim] if metric["name"] == "SVM"][0][
                f"{metric_name}_train"
            ]
        )
        train_metric_knn.append(
            [metric for metric in metrics[emb_dim] if metric["name"] == "KNN"][0][
                f"{metric_name}_train"
            ]
        )
        train_metric_rf.append(
            [metric for metric in metrics[emb_dim] if metric["name"] == "RF"][0][
                f"{metric_name}_train"
            ]
        )

    # plot with plotly
    metric_fig = go.Figure()
    metric_fig.add_trace(
        go.Scatter(
            x=list(metrics.keys()), y=metric_svm, mode="lines+markers", name="SVM"
        )
    )
    metric_fig.add_trace(
        go.Scatter(
            x=list(metrics.keys()), y=metric_knn, mode="lines+markers", name="KNN"
        )
    )
    metric_fig.add_trace(
        go.Scatter(x=list(metrics.keys()), y=metric_rf, mode="lines+markers", name="RF")
    )
    metric_fig.add_trace(
        go.Scatter(
            x=list(metrics.keys()),
            y=train_metric_svm,
            mode="lines+markers",
            name="SVM train",
            line=dict(dash="dash"),
        )
    )
    metric_fig.add_trace(
        go.Scatter(
            x=list(metrics.keys()),
            y=train_metric_knn,
            mode="lines+markers",
            name="KNN train",
            line=dict(dash="dash"),
        )
    )
    metric_fig.add_trace(
        go.Scatter(
            x=list(metrics.keys()),
            y=train_metric_rf,
            mode="lines+markers",
            name="RF train",
            line=dict(dash="dash"),
        )
    )
    metric_fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    # metric_fig.show()

    if filename:
        # if filename folder not exist, create it
        filename.parent.mkdir(parents=True, exist_ok=True)
        # save the figure
        metric_fig.write_image(filename)
