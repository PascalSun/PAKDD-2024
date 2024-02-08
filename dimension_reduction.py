from pathlib import Path

import pandas as pd

from models.TabularEncoder import TabularEncoder
from utils.logger import get_logger
from utils.plt_tsne import plot_tsne
from utils.timer import timer

logger = get_logger()


def dimension_reduction(
        arguments,
        dataset,
        embedding_df: pd.DataFrame,
        label_dict: dict,
        report_dir: Path,
):
    """
    Dimension reduction for the features
    Parameters
    ----------
    arguments
    dataset
    embedding_df
    label_dict
    report_dir

    Returns
    -------

    """
    if not report_dir.exists():
        report_dir.mkdir(parents=True)

    dataset.data.to(arguments.device)
    dr_auto_encoder = TabularEncoder(
        dataset.data.x,
        hidden_dims=[500, 250, 150],
        encoding_dim=50,
        device=arguments.device,
    )
    dr_auto_encoder.train()
    dr_x = (
        dr_auto_encoder.autoencoder.encoder(dataset.data.x).detach().cpu().numpy()
    )  # noqa
    dr_feature_df = pd.DataFrame(
        data=dr_x,
    )
    # for plot purpose
    dr_feature_df_tsne = dr_feature_df.copy(deep=True)
    dr_feature_df_tsne["y"] = dataset.data.y.cpu().numpy()
    with timer(logger, "Plotting DR TSNE"):
        plot_tsne(
            emb_df=dr_feature_df_tsne,
            title="DR TSNE",
            filename=report_dir / "10-dr.png",
            label_int_2_str=label_dict,
        )
    embedding_df = pd.concat([embedding_df, dr_feature_df], axis=1)

    return embedding_df
