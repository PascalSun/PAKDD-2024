import argparse
from typing import List, Optional

from src.iid.utils.constants import (
    GAEEncoderEnum,
    GAEFeatureEnum,
    GraphSAGEAggrEnum,
    ModelsEnum,
    NNTypeEnum,
    Node2VecParamModeEnum,
)


def parse_args(arguments: Optional[List[str]] = None):
    """
    Parse the arguments.
    :param arguments:
    :return:
    """
    parser = argparse.ArgumentParser(description="PhD IID")
    parser.add_argument(
        "-d", "--dataset", default="Cora", type=str, help="dataset name"
    )
    # for travel dataset
    parser.add_argument(
        "-travel_dataset",
        "--travel_dataset",
        default="Miami_FL",
        type=str,
        help="travel dataset name",
    )
    parser.add_argument(
        "-travel_task",
        "--travel_task",
        default="severity",
        type=str,
        help="travel dataset task, can be occur or severity",
    )

    parser.add_argument(
        "-m",
        "--model",
        default=ModelsEnum.feature_centrality.value,
        type=str,
        choices=[model.value for model in ModelsEnum],
        help="model name",
    )
    parser.add_argument("-e", "--epochs", default=50, type=int, help="number of epochs")

    parser.add_argument(
        "-device", "--device", default="cuda", type=str, help="device name"
    )

    # for oversampling, how to do that
    parser.add_argument(
        "-oversampling",
        "--oversampling",
        default="smote",
        type=str,
        choices=["smote", "random", "none"],
        help="oversampling",
    )

    # for whether plot the confusion matrix in PyCharm
    parser.add_argument("--plot_it", dest="plot_it", action="store_true")
    parser.add_argument("--no_plot_it", dest="plot_it", action="store_false")
    parser.set_defaults(plot_it=False)

    # for whether doing grid search
    parser.add_argument(
        "-grid_search", "--grid_search", dest="grid_search", action="store_true"
    )
    parser.add_argument(
        "-no_grid_search", "--no_grid_search", dest="grid_search", action="store_false"
    )
    parser.set_defaults(grid_search=False)

    # for unbalanced dataset, balanced accuracy
    parser.add_argument(
        "-balanced_ac",
        "--balanced_ac",
        dest="balanced_ac",
        action="store_true",
    )
    parser.add_argument(
        "-no_balanced_ac",
        "--no_balanced_ac",
        dest="balanced_ac",
        action="store_false",
    )
    parser.set_defaults(balanced_ac=False)

    # for dimensionality test
    parser.add_argument(
        "-start_dim",
        "--start_dim",
        default=8,
        type=int,
        help="start or default dimensionality of the embedding",
    )
    parser.add_argument(
        "-end_dim",
        "--end_dim",
        default=20,
        type=int,
        help="end dimensionality of the embedding",
    )

    # for graph_sage
    parser.add_argument(
        "-graph_sage_type",
        "--graph_sage_type",
        default=NNTypeEnum.unsupervised.value,
        type=str,
        choices=[graph_sage_type.value for graph_sage_type in NNTypeEnum],
        help="graph sage type, can be supervised or unsupervised_centrality, unsupervised_features",
    )
    parser.add_argument(
        "-graph_sage_hidden_dim",
        "--graph_sage_hidden_dim",
        default="16",
        type=str,
        help="hidden dimension of the graph sage model, input will be like '32,16'",
    )
    parser.add_argument(
        "-graph_sage_aggr",
        "--graph_sage_aggr",
        default=GraphSAGEAggrEnum.max.value,
        type=str,
        choices=[aggr.value for aggr in GraphSAGEAggrEnum],
        help="aggregation method of the graph sage model",
    )
    parser.add_argument(
        "-graph_sage_dataset_type",
        "--graph_sage_dataset_type",
        default=GAEFeatureEnum.feature_1433.value,
        type=str,
        choices=[dataset_type.value for dataset_type in GAEFeatureEnum],
        help="when training the unsupervised model, which type of dataset to use, with features or just centrality",
    )

    # for gcn
    parser.add_argument(
        "-gcn_type",
        "--gcn_type",
        default=NNTypeEnum.unsupervised.value,
        type=str,
        choices=[graph_sage_type.value for graph_sage_type in NNTypeEnum],
        help="graph sage type, can be supervised or unsupervised_centrality, unsupervised_features",
    )
    parser.add_argument(
        "-gcn_hidden_dim",
        "--gcn_hidden_dim",
        default="16",
        type=str,
        help="hidden dimension of the gcn model, input will be like '32,16'",
    )
    parser.add_argument(
        "-gcn_dataset_type",
        "--gcn_dataset_type",
        default=GAEFeatureEnum.feature_1433.value,
        type=str,
        choices=[dataset_type.value for dataset_type in GAEFeatureEnum],
        help="when training the unsupervised model, which type of dataset to use, with features or just centrality",
    )

    # for node2vec
    parser.add_argument(
        "-node2vec_params_mode",
        "--node2vec_params_mode",
        default=Node2VecParamModeEnum.dim.value,
        type=str,
        choices=[param_mode.value for param_mode in Node2VecParamModeEnum],
        help="node2vec params mode, can be dim or walk_length",
    )
    parser.add_argument(
        "-node2vec_walk_length",
        "--node2vec_walk_length",
        default=5,
        type=int,
        help="walk length of the node2vec model",
    )
    parser.add_argument(
        "-node2vec_context_size",
        "--node2vec_context_size",
        default=5,
        type=int,
        help="context size of the node2vec model",
    )
    parser.add_argument(
        "-node2vec_walks_per_node",
        "--node2vec_walks_per_node",
        default=2,
        type=int,
        help="walks per node of the node2vec model",
    )
    parser.add_argument(
        "-node2vec_num_negative_samples",
        "--node2vec_num_negative_samples",
        default=1,
        type=int,
        help="num negative samples of the node2vec model",
    )
    parser.add_argument(
        "-node2vec_p",
        "--node2vec_p",
        default=1.0,
        type=float,
        help="p of the node2vec model",
    )
    parser.add_argument(
        "-node2vec_q",
        "--node2vec_q",
        default=1.0,
        type=float,
        help="q of the node2vec model",
    )
    parser.add_argument(
        "-node2vec_pq_step",
        "--node2vec_pq_step",
        default=0.1,
        type=float,
        help="p q value step step of the node2vec model",
    )

    # for GAE
    parser.add_argument(
        "-gae_encoder",
        "--gae_encoder",
        default=GAEEncoderEnum.gcn.value,
        type=str,
        choices=[gae_encoder.value for gae_encoder in GAEEncoderEnum],
        help="encoder of the GAE model",
    )
    parser.add_argument(
        "-gae_feature",
        "--gae_feature",
        default=GAEFeatureEnum.centrality.value,
        type=str,
        choices=[gae_feature.value for gae_feature in GAEFeatureEnum],
        help="use which feature as input",
    )

    # for dimension reduction
    parser.add_argument(
        "-dim_reduction",
        "--dim_reduction",
        dest="dim_reduction",
        action="store_true",
    )
    parser.add_argument(
        "-no_dim_reduction",
        "--no_dim_reduction",
        dest="dim_reduction",
        action="store_false",
    )
    parser.set_defaults(dim_reduction=False)

    return parser.parse_args(arguments)
