import argparse
import copy
import pickle
import random
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa
from torch.nn import Embedding, init
from torch_geometric.nn import GAE  # noqa Graph Autoencoder
from torch_geometric.nn import GCNConv  # noqa
from torch_geometric.nn import Node2Vec  # noqa
from torch_geometric.nn import SAGEConv  # noqa
from torch_geometric.utils import to_networkx

from src.iid.dataset.load_datasets import load_dataset
from src.iid.dataset.travel import TRAVELDataset
from src.iid.dimension_reduction import dimension_reduction
from src.iid.models.GAE import train_gae
from src.iid.models.GCN import GCNEmb, GCNTask
from src.iid.models.GRAPH_AG import GraphAGEmb
from src.iid.models.GRAPH_SAGE import GraphSAGEEmb, GraphSAGETask
from src.iid.models.ML import GraphMLTrain
from src.iid.utils import (
    convert_networkx_to_torch_graph_with_centrality_features,
    evaluate_model,
    plot_metrics,
)
from src.iid.utils.constants import (
    DataSetEnum,
    DataSetModel,
    GAEEncoderEnum,
    GAEFeatureEnum,
    MLDefaultSettings,
    ModelsEnum,
    NNTypeEnum,
    Node2VecParamModeEnum,
    TravelDatasetName,
)
from src.iid.utils.split_data import split_data
from src.utils.constants import DATA_DIR, REPORT_DIR
from src.utils.logger import get_logger
from src.utils.plt_tsne import plot_tsne
from src.utils.timer import timer
from src.utils.to_json_file import to_json_file

# set seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

logger = get_logger()


def process(arguments: argparse.Namespace):
    # initialize
    data_dir, report_dir, device, ml_default_settings = init_setup(arguments)

    # load dataset and preprocess
    (
        dataset,
        dataset_with_centrality,
        label_list,
        label_dict,
    ) = load_and_preprocess_data(data_dir, arguments.dataset, arguments)

    if arguments.model == ModelsEnum.feature_centrality.value:
        metrics = handle_feature_centrality_model(
            arguments,
            ml_default_settings=ml_default_settings,
            dataset_with_centrality=dataset_with_centrality,
            report_dir=report_dir,
            label_list=label_list,
        )
    elif arguments.model == ModelsEnum.feature_1433.value:
        metrics = handle_feature_1433_model(
            arguments,
            ml_default_settings=ml_default_settings,
            dataset=dataset,
            report_dir=report_dir,
            label_list=label_list,
        )
    elif arguments.model == ModelsEnum.random_input.value:
        metrics = handle_random_input_model(
            arguments,
            ml_default_settings=ml_default_settings,
            dataset=dataset,
            report_dir=report_dir,
            label_list=label_list,
        )
    elif arguments.model == ModelsEnum.graph_sage.value:
        metrics = handle_graph_sage_model(
            arguments,
            ml_default_settings=ml_default_settings,
            dataset=dataset,
            dataset_with_centrality=dataset_with_centrality,
            report_dir=report_dir,
            label_list=label_list,
            label_dict=label_dict,
            device=device,
        )
    elif arguments.model == ModelsEnum.gcn.value:
        metrics = handle_gcn_model(
            arguments,
            ml_default_settings=ml_default_settings,
            dataset=dataset,
            dataset_with_centrality=dataset_with_centrality,
            report_dir=report_dir,
            label_list=label_list,
            label_dict=label_dict,
            device=device,
        )
    elif arguments.model == ModelsEnum.node2vec.value:
        metrics = handle_node2vec_model(
            arguments,
            ml_default_settings=ml_default_settings,
            dataset=dataset,
            report_dir=report_dir,
            label_list=label_list,
            label_dict=label_dict,
            device=device,
        )
    elif arguments.model == ModelsEnum.gae.value:
        metrics = handle_gae_model(
            arguments,
            ml_default_settings=ml_default_settings,
            dataset=dataset,
            dataset_with_centrality=dataset_with_centrality,
            report_dir=report_dir,
            label_list=label_list,
            label_dict=label_dict,
            device=device,
        )
    elif arguments.model == ModelsEnum.graph_ag.value:
        metrics = handle_graph_ag_model(
            arguments,
            ml_default_settings=ml_default_settings,
            dataset=dataset,
            # dataset_with_centrality=dataset_with_centrality,
            report_dir=report_dir,
            label_list=label_list,
            label_dict=label_dict,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model: {arguments.model}")

    return metrics


def init_setup(arguments: argparse.Namespace):
    DataSetModel(
        dataset=arguments.dataset,
    )
    data_dir = DATA_DIR / arguments.dataset
    report_dir = REPORT_DIR / arguments.dataset
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    if not report_dir.exists():
        report_dir.mkdir(parents=True)

    # if it is travel, then report should be added: / dataset.name
    if arguments.dataset.lower() == DataSetEnum.Travel.value.lower():
        report_dir = report_dir / arguments.travel_dataset
        if not report_dir.exists():
            report_dir.mkdir(parents=True)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and arguments.device == "cuda") else "cpu"
    )
    logger.info(f"Device: {device}")

    # params
    ml_default_settings = MLDefaultSettings()

    return data_dir, report_dir, device, ml_default_settings


def load_and_preprocess_data(data_dir, dataset_name, arguments):  # noqa
    # dataset
    if dataset_name.lower() == DataSetEnum.Travel.value.lower():
        # verify the arguments.travel_dataset
        if len(arguments.travel_dataset.split("_")) == 1:
            city = None
            state = arguments.travel_dataset
        else:
            city, state = arguments.travel_dataset.split("_")
        TravelDatasetName(state=state, city=city)

        dataset = TRAVELDataset(
            root=data_dir, name=arguments.travel_dataset, task=arguments.travel_task
        )
        dataset.data.labels = {i: str(i) for i in enumerate(dataset.data.y)}

    elif dataset_name.lower() == DataSetEnum.AttributedGraphDataset_PPI.value.lower():
        dataset = load_dataset(dataset_name)
        # TODO: this is a multi-class task
        dataset.data.labels = {i: str(i) for i in enumerate(dataset.data.y.tolist())}
    else:
        dataset = load_dataset(dataset_name)

    if (
        arguments.dataset.lower()
        == DataSetEnum.AttributedGraphDataset_Flickr.value.lower()
    ):
        dataset.data.x = dataset.data.x.to_dense()
    # for the dataset with test/validation/train split
    if dataset_name.lower() in [
        DataSetEnum.KarateClub.value.lower(),
        DataSetEnum.AMAZON_COMPUTERS.value.lower(),
        DataSetEnum.AMAZON_PHOTO.value.lower(),
        DataSetEnum.TWITCH_DE.value.lower(),
        DataSetEnum.TWITCH_EN.value.lower(),
        DataSetEnum.TWITCH_ES.value.lower(),
        DataSetEnum.TWITCH_FR.value.lower(),
        DataSetEnum.TWITCH_PT.value.lower(),
        DataSetEnum.TWITCH_RU.value.lower(),
        DataSetEnum.GitHub.value.lower(),
        DataSetEnum.Coauthor_CS.value.lower(),
        DataSetEnum.Coauthor_Physics.value.lower(),
        DataSetEnum.CitationsFull_Cora.value.lower(),
        DataSetEnum.CitationsFull_CiteSeer.value.lower(),
        DataSetEnum.CitationsFull_PubMed.value.lower(),
        DataSetEnum.CitationsFull_Cora_ML.value.lower(),
        DataSetEnum.CitationsFull_DBLP.value.lower(),
        DataSetEnum.Cora_Full.value.lower(),
        DataSetEnum.WEBKB_Cornell.value.lower(),
        DataSetEnum.WEBKB_Texas.value.lower(),
        DataSetEnum.WEBKB_Wisconsin.value.lower(),
        DataSetEnum.PolBlogs.value.lower(),
        DataSetEnum.EllipticBitcoinDataset.value.lower(),
        DataSetEnum.AttributedGraphDataset_Cora.value.lower(),
        DataSetEnum.TWITCH_FR.value.lower(),
        DataSetEnum.AttributedGraphDataset_CiteSeer.value.lower(),
        DataSetEnum.AttributedGraphDataset_Pubmed.value.lower(),
        DataSetEnum.AttributedGraphDataset_BlogCatalog.value.lower(),
        DataSetEnum.Actor.value.lower(),
        DataSetEnum.AttributedGraphDataset_PPI.value.lower(),
        DataSetEnum.AttributedGraphDataset_Wiki.value.lower(),
        DataSetEnum.AttributedGraphDataset_Flickr.value.lower(),
        DataSetEnum.HeterophilousGraphDataset_Amazon_ratings.value.lower(),
        DataSetEnum.HeterophilousGraphDataset_Questions.value.lower(),
        DataSetEnum.HeterophilousGraphDataset_Tolokers.value.lower(),
        DataSetEnum.HeterophilousGraphDataset_Roman_empire.value.lower(),
        DataSetEnum.HeterophilousGraphDataset_Minesweeper.value.lower(),
    ]:
        dataset = split_data(dataset)
    dataset.name = dataset_name
    label_dict = dataset.data.labels
    label_list = [value for key, value in label_dict.items()]
    logger.info(f"Label list: {label_list}")

    dataset_graph = to_networkx(
        dataset.data,
        to_undirected=True,
        node_attrs=["y", "train_mask", "val_mask", "test_mask"],
    )

    # preprocess the graph to generate a data with centrality features
    with timer(
        logger, "Preprocess the graph to generate a data with centrality features"
    ):
        graph_metrics_dir = data_dir / dataset.name / "graph_metrics"
        if not graph_metrics_dir.exists():
            graph_metrics_dir.mkdir(parents=True, exist_ok=True)
        with timer(logger, "Degree centrality"):
            if not (graph_metrics_dir / "degree.pickle").exists():
                degree = nx.degree_centrality(dataset_graph)

                # Normalize the degree centrality values
                max_degree = max(degree.values())
                min_degree = min(degree.values())
                for node, value in degree.items():
                    degree[node] = (value - min_degree) / (max_degree - min_degree)

                with open(graph_metrics_dir / "degree.pickle", "wb") as handle:
                    pickle.dump(degree, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(graph_metrics_dir / "degree.pickle", "rb") as handle:
                    degree = pickle.load(handle)

        with timer(logger, "Betweenness centrality"):
            if not (graph_metrics_dir / "betweenness.pickle").exists():
                betweenness = nx.betweenness_centrality(dataset_graph, k=5)

                # Normalize the betweenness centrality values
                max_betweenness = max(betweenness.values())
                min_betweenness = min(betweenness.values())
                for node, value in betweenness.items():
                    if max_betweenness == min_betweenness:
                        betweenness[node] = 0
                    else:
                        betweenness[node] = (value - min_betweenness) / (
                            max_betweenness - min_betweenness
                        )

                with open(graph_metrics_dir / "betweenness.pickle", "wb") as handle:
                    pickle.dump(betweenness, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(graph_metrics_dir / "betweenness.pickle", "rb") as handle:
                    betweenness = pickle.load(handle)

    data_with_centrality = convert_networkx_to_torch_graph_with_centrality_features(
        nx_graph=dataset_graph,
        degree_df=degree,
        betweenness_df=betweenness,
        y_field="y",
    )
    dataset_with_centrality = copy.deepcopy(dataset)
    dataset_with_centrality.data = data_with_centrality
    return dataset, dataset_with_centrality, label_list, label_dict


def handle_feature_centrality_model(
    arguments,
    ml_default_settings,
    dataset_with_centrality,
    report_dir,
    label_list,
):
    logger.info("Feature centrality model")
    feature_centrality_df = pd.DataFrame(dataset_with_centrality.data.x.numpy())
    feature_centrality_df["y"] = dataset_with_centrality.data.y.numpy()
    feature_centrality_model = GraphMLTrain(
        feature_centrality_df,
        y_field_name="y",
        svm_best_param=ml_default_settings.pretrain_svm_best_param,
        knn_best_param=ml_default_settings.pretrain_knn_best_param,
        rf_best_param=ml_default_settings.pretrain_rf_best_param,
        oversampling=arguments.oversampling,
        plot_it=arguments.plot_it,
        risk_labels=label_list,
    )
    feature_centrality_ml_models = feature_centrality_model.train(
        grid_search=arguments.grid_search
    )
    feature_centrality_model_metrics = feature_centrality_model.plt_3_confusion_matrix(
        feature_centrality_ml_models,
        f"{arguments.model.upper()} Confusion Matrix",
        report_dir / f"{arguments.model}.png",
        plot_it=arguments.plot_it,
        balanced_ac=arguments.balanced_ac,
    )
    logger.critical(feature_centrality_model_metrics)
    to_json_file(
        feature_centrality_model_metrics,
        report_dir / f"{arguments.model}_metrics.json",
    )
    return feature_centrality_model_metrics


def handle_feature_1433_model(
    arguments, ml_default_settings, dataset, report_dir, label_list
):
    logger.info("Feature 1433 model")
    try:
        feature_1433_df = pd.DataFrame(dataset.data.x.numpy())
    except Exception as e:
        logger.warning(e)
        feature_1433_df = pd.DataFrame(dataset.data.x.to_dense().numpy())
    feature_1433_df["y"] = dataset.data.y.numpy()
    feature_1433_model = GraphMLTrain(
        feature_1433_df,
        y_field_name="y",
        svm_best_param=ml_default_settings.pretrain_svm_best_param,
        knn_best_param=ml_default_settings.pretrain_knn_best_param,
        rf_best_param=ml_default_settings.pretrain_rf_best_param,
        oversampling=arguments.oversampling,
        plot_it=arguments.plot_it,
        risk_labels=label_list,
    )
    feature_1433_ml_models = feature_1433_model.train(grid_search=arguments.grid_search)
    feature_1433_model_metrics = feature_1433_model.plt_3_confusion_matrix(
        feature_1433_ml_models,
        f"{arguments.model.upper()} Confusion Matrix",
        report_dir / f"{arguments.model}.png",
        plot_it=arguments.plot_it,
        balanced_ac=arguments.balanced_ac,
    )
    logger.critical(feature_1433_model_metrics)
    to_json_file(
        feature_1433_model_metrics, report_dir / f"{arguments.model}_metrics.json"
    )
    return feature_1433_model_metrics


def handle_random_input_model(
    arguments, ml_default_settings, dataset, report_dir, label_list
):
    logger.info("Random input model")
    if arguments.start_dim > arguments.end_dim:
        raise ValueError(
            "Start dimensionality cannot be greater than end dimensionality"
        )

    random_input_model_performances = {}
    for random_dim in range(arguments.start_dim, arguments.end_dim + 1):
        random_embeddings = Embedding(dataset.data.y.shape[0], random_dim)
        # Initialize the embedding with random values
        init.xavier_uniform_(random_embeddings.weight.data)
        dataset.data.x = torch.tensor(random_embeddings.weight.data)

        x_df = pd.DataFrame(
            dataset.data.x.detach().cpu().numpy(), columns=list(range(random_dim))
        )
        x_df["y"] = dataset.data.y.detach().cpu().numpy()
        random_model = GraphMLTrain(
            x_df,
            y_field_name="y",
            svm_best_param=ml_default_settings.pretrain_svm_best_param,
            knn_best_param=ml_default_settings.pretrain_knn_best_param,
            rf_best_param=ml_default_settings.pretrain_rf_best_param,
            oversampling=arguments.oversampling,
            plot_it=arguments.plot_it,
            risk_labels=label_list,
        )
        random_train_models = random_model.train(grid_search=arguments.grid_search)
        random_model_metrics = random_model.plt_3_confusion_matrix(
            random_train_models,
            f"{arguments.model.upper()} Confusion Matrix",
            plot_it=arguments.plot_it,
            balanced_ac=arguments.balanced_ac,
        )
        random_input_model_performances[random_dim] = random_model_metrics

    for metric_name in [
        "accuracy_score",
        "macro_precision",
        "macro_recall",
        "macro_f_beta_score",
        "micro_precision",
        "micro_recall",
        "micro_f_beta_score",
        "roc_auc",
    ]:
        plot_metrics(
            metrics=random_input_model_performances,
            title="Random Input Model Performance vs Dimensionality",
            x_title="Dimensionality",
            y_title=metric_name.upper(),
            metric_name=metric_name,
            filename=report_dir
            / arguments.model
            / f"{arguments.model}-dim-{metric_name}.png",
        )

    to_json_file(
        random_input_model_performances,
        report_dir / arguments.model / f"{arguments.model}_metrics.json",
    )
    return random_input_model_performances


def handle_graph_sage_model(
    arguments,
    ml_default_settings,
    dataset,
    dataset_with_centrality,
    report_dir,
    label_list,
    label_dict,
    device,
):
    """
    We will have three different models for GraphSAGE:
    - supervised learning: centrality as features
    - supervised learning: 1433 features
    - unsupervised learning: reconstruct the graph as loss function
    """
    logger.info("GraphSAGE model")
    # supervised learning: centrality as features
    graph_sage_hidden_dim = [
        int(dim) for dim in arguments.graph_sage_hidden_dim.split(",")
    ]
    if arguments.graph_sage_type == NNTypeEnum.supervised_centrality.value:
        logger.info("GraphSAGE Classification with betweenness and degree")
        graph_sage_with_centrality_model = GraphSAGETask(
            dataset_with_centrality.num_node_features,
            graph_sage_hidden_dim,
            dataset_with_centrality.num_classes,
            aggr=arguments.graph_sage_aggr,
        ).to(device)
        dataset_with_centrality.data.to(device)

        graph_sage_with_centrality_model.fit(
            dataset_with_centrality.data, arguments.epochs
        )
        graph_sage_with_centrality_metrics = evaluate_model(
            graph_sage_with_centrality_model,
            dataset_with_centrality,
            title="GraphSAGE Classification with betweenness and degree",
            label_dict=label_dict,
            plot_it=arguments.plot_it,
            filename=report_dir / arguments.model / f"{arguments.model}-centrality.png",
        )
        to_json_file(
            graph_sage_with_centrality_metrics,
            report_dir / arguments.model / f"{arguments.model}-centrality-metrics.json",
        )
        return graph_sage_with_centrality_metrics

    # supervised learning: 1433 features
    if arguments.graph_sage_type == NNTypeEnum.supervised_feature.value:
        logger.info("GraphSAGE Classification with 1433 features")
        graph_sage_with_feature_model = GraphSAGETask(
            dataset.num_node_features,
            graph_sage_hidden_dim,
            dataset.num_classes,
            aggr=arguments.graph_sage_aggr,
        ).to(device)
        dataset.data.to(device)
        graph_sage_with_feature_model.fit(dataset.data, arguments.epochs)
        graph_sage_with_feature_metrics = evaluate_model(
            graph_sage_with_feature_model,
            dataset,
            title="GraphSAGE Classification with 1433 features",
            label_dict=label_dict,
            plot_it=arguments.plot_it,
            filename=report_dir / arguments.model / f"{arguments.model}-feature.png",
        )
        to_json_file(
            graph_sage_with_feature_metrics,
            report_dir / arguments.model / f"{arguments.model}-feature-metrics.json",
        )
        return graph_sage_with_feature_metrics

    # unsupervised learning: reconstruct the graph as loss function
    if arguments.graph_sage_type == NNTypeEnum.unsupervised.value:
        """
        Adjustable parameters:
        - dataset
        - dim: start_dim, end_dim
        - hidden_dim and layers
        """
        logger.info(
            f"GraphSAGE Unsupervised Learning with hidden_layers: {graph_sage_hidden_dim}"
        )

        # determine which dataset to use
        if arguments.graph_sage_dataset_type == GAEFeatureEnum.feature_1433.value:
            logger.info("Using 1433 features")
            graph_sage_dataset = dataset
        elif arguments.graph_sage_dataset_type == GAEFeatureEnum.centrality.value:
            logger.info("Using centrality features")
            graph_sage_dataset = dataset_with_centrality
        else:
            raise ValueError(
                f"Invalid graph_sage_dataset_type: {arguments.graph_sage_dataset_type}"
            )

        graph_sage_unsupervised_performance = {}
        for emb_dim in range(arguments.start_dim, arguments.end_dim + 1):
            graph_sage_emb_params = dict(
                in_channels=graph_sage_dataset.num_node_features,
                hidden_channels=graph_sage_hidden_dim,
                out_channels=emb_dim,
                aggr=arguments.graph_sage_aggr,
            )
            graph_sage_emb_model = GraphSAGEEmb(**graph_sage_emb_params)
            logger.info(graph_sage_emb_params)

            graph_sage_dataset.data.to(device)
            graph_sage_emb_model.to(device)
            graph_sage_emb_df = graph_sage_emb_model.fit(
                graph_sage_dataset.data, arguments.epochs
            )

            plot_tsne(
                graph_sage_emb_df,
                title=f"GraphSAGE Unsupervised Learning with {emb_dim} dimensions",
                filename=report_dir
                / arguments.model
                / arguments.graph_sage_dataset_type
                / str(len(graph_sage_hidden_dim))
                / f"{arguments.model}-dim-{emb_dim}.png",
            )
            graph_sage_emb_model = GraphMLTrain(
                graph_sage_emb_df,
                y_field_name="y",
                svm_best_param=ml_default_settings.pretrain_svm_best_param,
                knn_best_param=ml_default_settings.pretrain_knn_best_param,
                rf_best_param=ml_default_settings.pretrain_rf_best_param,
                oversampling=arguments.oversampling,
                plot_it=arguments.plot_it,
                risk_labels=label_list,
            )
            graph_sage_emb_train_models = graph_sage_emb_model.train(
                grid_search=arguments.grid_search
            )
            graph_sage_emb_model_metrics = graph_sage_emb_model.plt_3_confusion_matrix(
                graph_sage_emb_train_models,
                f"{arguments.model.upper()} Confusion Matrix",
                plot_it=arguments.plot_it,
                balanced_ac=arguments.balanced_ac,
            )
            graph_sage_unsupervised_performance[emb_dim] = graph_sage_emb_model_metrics
        graph_title = f"GraphSAGE Model: {arguments.graph_sage_dataset_type.upper()}/Hidden:{len(graph_sage_hidden_dim)} Performance vs Dimensionality"  # noqa
        for metric_name in [
            "accuracy_score",
            "macro_precision",
            "macro_recall",
            "macro_f_beta_score",
            "micro_precision",
            "micro_recall",
            "micro_f_beta_score",
            "roc_auc",
        ]:
            plot_metrics(
                metrics=graph_sage_unsupervised_performance,
                title=graph_title,
                x_title="Dimensionality",
                y_title=metric_name.upper(),
                metric_name=metric_name,
                filename=report_dir
                / arguments.model
                / arguments.graph_sage_dataset_type
                / str(len(graph_sage_hidden_dim))
                / f"{arguments.model}-dim-{metric_name}.png",
            )
        to_json_file(
            graph_sage_unsupervised_performance,
            report_dir
            / arguments.model
            / arguments.graph_sage_dataset_type
            / str(len(graph_sage_hidden_dim))
            / f"{arguments.model}_metrics.json",
        )
        return graph_sage_unsupervised_performance


def handle_gcn_model(
    arguments: argparse.Namespace,
    ml_default_settings,
    dataset,
    dataset_with_centrality,
    report_dir,
    label_list,
    label_dict,
    device,
):
    """
    GCN Classification
    - supervised learning: centrality as features
    - supervised learning: 1433 features
    - unsupervised learning: reconstruct the graph as loss function
    """
    logger.info(f"GCN Model: {arguments.gcn_type}")
    gcn_hidden_dim = [int(x) for x in arguments.gcn_hidden_dim.split(",")]
    if arguments.gcn_type == NNTypeEnum.supervised_centrality.value:
        logger.info("GCN Classification with centrality as features")
        gcn_centrality_model = GCNTask(
            dataset_with_centrality.num_node_features,
            gcn_hidden_dim,
            dataset_with_centrality.num_classes,
        ).to(device)
        dataset_with_centrality.data.to(device)
        gcn_centrality_model.fit(dataset_with_centrality.data, arguments.epochs)
        gcn_centrality_metrics = evaluate_model(
            gcn_centrality_model,
            dataset_with_centrality,
            title="GCN Classification",
            label_dict=label_dict,
            plot_it=arguments.plot_it,
            filename=report_dir / arguments.model / f"{arguments.model}-centrality.png",
        )
        to_json_file(
            gcn_centrality_metrics,
            report_dir / arguments.model / f"{arguments.model}-centrality-metrics.json",
        )
        return gcn_centrality_metrics
    if arguments.gcn_type == NNTypeEnum.supervised_feature.value:
        logger.info("GCN Classification with 1433 features")
        gcn_feature_model = GCNTask(
            dataset.num_node_features, gcn_hidden_dim, dataset.num_classes
        ).to(device)
        dataset.data.to(device)
        gcn_feature_model.fit(dataset.data, arguments.epochs)
        gcn_feature_metrics = evaluate_model(
            gcn_feature_model,
            dataset,
            title="GCN Classification",
            label_dict=label_dict,
            plot_it=arguments.plot_it,
            filename=report_dir / arguments.model / f"{arguments.model}-feature.png",
        )
        to_json_file(
            gcn_feature_metrics,
            report_dir / arguments.model / f"{arguments.model}-feature-metrics.json",
        )
        return gcn_feature_metrics

    if arguments.gcn_type == NNTypeEnum.unsupervised.value:
        """
        Same here as GraphSAGE
        adjustable parameters:
        - dataset
        - hidden_dim and layers
        - dim: dimensionality of the embedding
        """
        logger.info("GCN Unsupervised Learning")

        if arguments.gcn_dataset_type == GAEFeatureEnum.feature_1433.value:
            logger.info("GCN Unsupervised Learning with 1433 features")
            gcn_dataset = dataset
        elif arguments.gcn_dataset_type == GAEFeatureEnum.centrality.value:
            logger.info("GCN Unsupervised Learning with centrality as features")
            gcn_dataset = dataset_with_centrality
        else:
            raise ValueError("Invalid GCN Dataset Type")

        gcn_unsupervised_performance = {}
        for emb_dim in range(arguments.start_dim, arguments.end_dim + 1):
            gcn_emb_model = GCNEmb(
                gcn_dataset.num_node_features, gcn_hidden_dim, emb_dim
            )
            gcn_dataset.data.to(device)
            gcn_emb_model.to(device)
            gcn_emb_df = gcn_emb_model.fit(gcn_dataset.data, arguments.epochs)

            # plot tsne
            plot_tsne(
                gcn_emb_df,
                title=f"GCN Unsupervised Learning with {emb_dim} dimensions",
                filename=report_dir
                / arguments.model
                / arguments.gcn_dataset_type
                / str(len(gcn_hidden_dim))
                / f"{arguments.model}-dim-{emb_dim}.png",
            )

            gcn_emb_model = GraphMLTrain(
                gcn_emb_df,
                y_field_name="y",
                svm_best_param=ml_default_settings.pretrain_svm_best_param,
                knn_best_param=ml_default_settings.pretrain_knn_best_param,
                rf_best_param=ml_default_settings.pretrain_rf_best_param,
                oversampling=arguments.oversampling,
                plot_it=arguments.plot_it,
                risk_labels=label_list,
            )
            gcn_emb_train_models = gcn_emb_model.train(
                grid_search=arguments.grid_search
            )
            gcn_emb_model_metrics = gcn_emb_model.plt_3_confusion_matrix(
                gcn_emb_train_models,
                f"{arguments.model.upper()} Confusion Matrix",
                plot_it=arguments.plot_it,
                balanced_ac=arguments.balanced_ac,
            )
            gcn_unsupervised_performance[emb_dim] = gcn_emb_model_metrics
        graph_title = f"GCN Model: {arguments.gcn_dataset_type.upper()}/Hidden:{len(gcn_hidden_dim)} Performance vs Dimensionality"  # noqa
        for metric_name in [
            "accuracy_score",
            "macro_precision",
            "macro_recall",
            "macro_f_beta_score",
            "micro_precision",
            "micro_recall",
            "micro_f_beta_score",
            "roc_auc",
        ]:
            plot_metrics(
                metrics=gcn_unsupervised_performance,
                title=graph_title,
                x_title="Dimensionality",
                y_title=metric_name.upper(),
                metric_name=metric_name,
                filename=report_dir
                / arguments.model
                / arguments.gcn_dataset_type
                / str(len(gcn_hidden_dim))
                / f"{arguments.model}-dim-{metric_name}.png",
            )
        to_json_file(
            gcn_unsupervised_performance,
            report_dir
            / arguments.model
            / arguments.gcn_dataset_type
            / str(len(gcn_hidden_dim))
            / f"{arguments.model}-dim-metrics.json",
        )
        return gcn_unsupervised_performance


def handle_node2vec_model(
    arguments: argparse.Namespace,
    ml_default_settings,
    dataset,
    report_dir,
    label_list,
    label_dict: dict,
    device: torch.device = torch.device("cpu"),
):
    # DISCUSS: why is this one so good?
    """
    Node2vec

    For some reason, this one performs really before, and now is very good now.
    It does not make sense to me, as this one is not include the 1433 features.

    My understanding is that, the graph already have the structure to make this 7 classifications into
    different clusters, so the random walk will gather this group information, and then transform that
    into the embedding space

    """

    if arguments.start_dim > arguments.end_dim:
        raise ValueError("Start dimensionality must be less than end dimensionality")
    node2vec_unsupervised_performance = {}

    node2vec_params = []

    param_mapping = {
        Node2VecParamModeEnum.dim.value: (
            "embedding_dim",
            range(arguments.start_dim, arguments.end_dim + 1),
        ),
        Node2VecParamModeEnum.walk_length.value: (
            "walk_length",
            range(2, arguments.node2vec_walk_length + 1),
        ),
        Node2VecParamModeEnum.walk_per_node.value: (
            "walks_per_node",
            range(1, arguments.node2vec_walks_per_node + 1),
        ),
        Node2VecParamModeEnum.num_negative_samples.value: (
            "num_negative_samples",
            range(1, arguments.node2vec_num_negative_samples + 1, 30),
        ),
        Node2VecParamModeEnum.p.value: (
            "p",
            np.array(
                range(1, int((arguments.node2vec_p + 1) / arguments.node2vec_pq_step))
            )
            * arguments.node2vec_pq_step,
        ),
        Node2VecParamModeEnum.q.value: (
            "q",
            np.array(
                range(1, int((arguments.node2vec_q + 1) / arguments.node2vec_pq_step))
            )
            * arguments.node2vec_pq_step,
        ),
    }
    if arguments.node2vec_params_mode not in param_mapping:
        raise ValueError(
            f"Invalid node2vec params mode: {arguments.node2vec_params_mode}"
        )

    node2vec_performance_key, param_range = param_mapping[
        arguments.node2vec_params_mode
    ]
    raw_params = dict(
        edge_index=dataset.data.edge_index,
        embedding_dim=arguments.start_dim,
        walk_length=arguments.node2vec_walk_length,
        context_size=arguments.node2vec_context_size,
        num_negative_samples=arguments.node2vec_num_negative_samples,
        p=arguments.node2vec_p,
        q=arguments.node2vec_q,
    )
    for param_value in param_range:
        generated_params = {
            **raw_params,
            node2vec_performance_key: param_value,
        }
        if arguments.node2vec_params_mode == Node2VecParamModeEnum.walk_length.value:
            generated_params["context_size"] = param_value
        node2vec_params.append(generated_params)

    for params in node2vec_params:
        logger.info(params)
        node2vec_model = Node2Vec(
            **params,
        )
        performance_key_value = params[node2vec_performance_key]
        node2vec_model.reset_parameters()
        node2vec_model.to(device)
        node2vec_optimizer = torch.optim.Adam(
            list(node2vec_model.parameters()), lr=0.01
        )
        node2vec_loader = node2vec_model.loader(
            batch_size=128, shuffle=True, num_workers=4
        )
        # training the node2vec
        for epoch in range(arguments.epochs + 1):
            node2vec_model.train()
            total_loss = 0
            for pos_rw, neg_rw in node2vec_loader:
                node2vec_optimizer.zero_grad()
                loss = node2vec_model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                node2vec_optimizer.step()
                total_loss += loss.item()

            with torch.no_grad():
                node2vec_model.eval()
                z = node2vec_model()
                # logger.info(z.shape)
                # logger.info(dataset.data.train_mask.shape)
                # the test is multiclass classification following ML tasks
                # but this does not affect the loss function and the training
                # so actually the logistic regression is already with good performance
                test_acc = node2vec_model.test(
                    z[dataset.data.train_mask],
                    dataset.data.y[dataset.data.train_mask],
                    z[dataset.data.test_mask],
                    dataset.data.y[dataset.data.test_mask],
                )
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch: {epoch:03d}, Loss: {total_loss:.4f}, Test: {test_acc:.4f}"
                )

        node2vec_embeddings = node2vec_model.embedding.weight.cpu().detach().numpy()
        node2vec_embeddings_df = pd.DataFrame(node2vec_embeddings)
        node2vec_embeddings_df["y"] = dataset.data.y.cpu().detach().numpy()

        plot_tsne(
            emb_df=node2vec_embeddings_df,
            title=f"{arguments.model.upper()}/{node2vec_performance_key.upper()}/{performance_key_value}",
            filename=report_dir
            / arguments.model
            / node2vec_performance_key
            / f"{arguments.model}-{node2vec_performance_key}-{performance_key_value}.png",
            label_int_2_str=label_dict,
        )
        # train classification
        if arguments.dim_reduction:
            node2vec_embeddings_df = dimension_reduction(
                arguments,
                dataset,
                embedding_df=node2vec_embeddings_df,
                label_dict=label_dict,
                report_dir=report_dir / arguments.model / "DR",
            )
        node2vec_ml_model = GraphMLTrain(
            node2vec_embeddings_df,
            y_field_name="y",
            svm_best_param=ml_default_settings.pretrain_svm_best_param,
            knn_best_param=ml_default_settings.pretrain_knn_best_param,
            rf_best_param=ml_default_settings.pretrain_rf_best_param,
            oversampling=arguments.oversampling,
            plot_it=arguments.plot_it,
            risk_labels=label_list,
        )
        node2vec_ml_models = node2vec_ml_model.train(grid_search=arguments.grid_search)
        node2vec_ml_model_metrics = node2vec_ml_model.plt_3_confusion_matrix(
            node2vec_ml_models,
            f"{arguments.model.upper()} Confusion Matrix",
            plot_it=arguments.plot_it,
            balanced_ac=arguments.balanced_ac,
        )
        node2vec_unsupervised_performance[
            performance_key_value
        ] = node2vec_ml_model_metrics

    for metric_name in [
        "accuracy_score",
        "macro_precision",
        "macro_recall",
        "macro_f_beta_score",
        "micro_precision",
        "micro_recall",
        "micro_f_beta_score",
        "roc_auc",
    ]:
        plot_metrics(
            metrics=node2vec_unsupervised_performance,
            title=f"Node2vec Embedding Model Performance vs {node2vec_performance_key.upper()}",
            x_title=node2vec_performance_key.upper(),
            y_title=metric_name.upper(),
            metric_name=metric_name,
            filename=report_dir
            / arguments.model
            / node2vec_performance_key
            / f"{arguments.model}-{node2vec_performance_key}-{metric_name}.png",
        )
    to_json_file(
        node2vec_unsupervised_performance,
        report_dir
        / arguments.model
        / node2vec_performance_key
        / f"{arguments.model}-{node2vec_performance_key}-metrics.json",
    )
    return node2vec_unsupervised_performance


def handle_gae_model(
    arguments: argparse.Namespace,
    ml_default_settings,
    dataset,
    dataset_with_centrality,
    report_dir,
    label_list,
    label_dict: Optional[dict] = None,
    device: torch.device = torch.device("cpu"),
):
    logger.info("GAE Training")
    if arguments.start_dim > arguments.end_dim:
        raise ValueError("Start dimensionality must be less than end dimensionality")

    if arguments.gae_encoder == GAEEncoderEnum.gcn.value:
        gcn_hidden_channels = [int(dim) for dim in arguments.gcn_hidden_dim.split(",")]
        gae_gcn_unsupervised_performance = {}
        logger.info("GAE Encoder: GCN")
        for emb_dim in range(arguments.start_dim, arguments.end_dim + 1):
            if arguments.gae_feature == GAEFeatureEnum.centrality.value:
                input_dataset = dataset_with_centrality
            elif arguments.gae_feature == GAEFeatureEnum.feature_1433.value:
                input_dataset = dataset
            else:
                raise ValueError("Invalid GAE Feature")

            gae_gcn_model = GCNEmb(
                in_channels=input_dataset.data.num_features,
                hidden_layer_dim=gcn_hidden_channels,
                out_channels=emb_dim,
            )
            gae_gcn_model.to(device)
            logger.info(device)
            input_dataset.data.to(device)
            gae_gcn_embeddings = train_gae(
                data=input_dataset.data,
                encoder=gae_gcn_model,
                device=device,
                epochs=arguments.epochs,
            )
            tsne_title = f"TSNE of {arguments.model.upper()}-{arguments.gae_encoder.upper()}-{arguments.gae_feature.upper()}-{emb_dim}"  # noqa

            with timer(logger, "Plotting TSNE"):
                plot_tsne(
                    emb_df=gae_gcn_embeddings,
                    title=tsne_title,
                    filename=report_dir
                    / arguments.model
                    / arguments.gae_encoder
                    / arguments.gae_feature
                    / f"{arguments.model}-{arguments.gae_encoder}-{arguments.gae_feature}-{emb_dim}.png",
                    label_int_2_str=label_dict,
                )

            # if we want to compress the embeddings
            if arguments.dim_reduction:
                gae_gcn_embeddings = dimension_reduction(
                    arguments,
                    dataset,
                    embedding_df=gae_gcn_embeddings,
                    label_dict=label_dict,
                    report_dir=report_dir
                    / arguments.model
                    / arguments.gae_encoder
                    / "DR",
                )
            # save the embedding to files
            logger.info("saving the csv")
            logger.critical(
                report_dir
                / arguments.model
                / arguments.gae_encoder
                / arguments.gae_feature
                / f"tsne-{arguments.model}-{arguments.gae_encoder}-{arguments.gae_feature}-{emb_dim}.csv"
            )
            # remove the y field
            gae_gcn_embeddings_csv = gae_gcn_embeddings.copy(deep=True)
            gae_gcn_embeddings_csv.drop(columns=["y"], inplace=True)
            gae_gcn_embeddings_csv.to_csv(
                report_dir
                / arguments.model
                / arguments.gae_encoder
                / arguments.gae_feature
                / f"tsne-{arguments.model}-{arguments.gae_encoder}-{arguments.gae_feature}-{emb_dim}.csv",
                index=False,
                header=False,
            )
            # train classification
            logger.info("Training traditional classification models")
            gae_ml_model = GraphMLTrain(
                gae_gcn_embeddings,
                y_field_name="y",
                svm_best_param=ml_default_settings.pretrain_svm_best_param,
                knn_best_param=ml_default_settings.pretrain_knn_best_param,
                rf_best_param=ml_default_settings.pretrain_rf_best_param,
                oversampling=arguments.oversampling,
                plot_it=arguments.plot_it,
                risk_labels=label_list,
            )
            with timer(logger, "Training traditional classification models"):
                gae_ml_models = gae_ml_model.train(grid_search=arguments.grid_search)
            gae_ml_model_metrics = gae_ml_model.plt_3_confusion_matrix(
                gae_ml_models,
                f"{arguments.model.upper()} Confusion Matrix",
                plot_it=arguments.plot_it,
                balanced_ac=arguments.balanced_ac,
            )
            gae_gcn_unsupervised_performance[emb_dim] = gae_ml_model_metrics
        for metric_name in [
            "accuracy_score",
            "macro_precision",
            "macro_recall",
            "macro_f_beta_score",
            "micro_precision",
            "micro_recall",
            "micro_f_beta_score",
            "roc_auc",
        ]:
            plot_metrics(
                metrics=gae_gcn_unsupervised_performance,
                title="GAE GCN Model Performance vs Dimensionality",
                x_title="Dimensionality",
                y_title=metric_name.upper(),
                metric_name=metric_name,
                filename=report_dir
                / arguments.model
                / arguments.gae_encoder
                / arguments.gae_feature
                / f"tsne-{arguments.model}-{arguments.gae_encoder}-{arguments.gae_feature}-dim-{metric_name}.png",
            )
        to_json_file(
            gae_gcn_unsupervised_performance,
            report_dir
            / arguments.model
            / arguments.gae_encoder
            / arguments.gae_feature
            / f"{arguments.model}-{arguments.gae_encoder}-{arguments.gae_feature}-dim-metrics.json",
        )
        return gae_gcn_unsupervised_performance

    if arguments.gae_encoder == GAEEncoderEnum.graph_sage.value:
        logger.info("GAE Encoder: GraphSAGE")
        graph_sage_hidden_channels = [
            int(dim) for dim in arguments.graph_sage_hidden_dim.split(",")
        ]
        gae_graph_sage_unsupervised_performance = {}
        for emb_dim in range(arguments.start_dim, arguments.end_dim + 1):
            if arguments.gae_feature == GAEFeatureEnum.centrality.value:
                input_dataset = dataset_with_centrality
            elif arguments.gae_feature == GAEFeatureEnum.feature_1433.value:
                input_dataset = dataset
            else:
                raise ValueError("Invalid GAE Feature")
            gae_graph_sage_model = GraphSAGEEmb(
                in_channels=input_dataset.data.num_features,
                hidden_channels=graph_sage_hidden_channels,
                out_channels=emb_dim,
                aggr=arguments.graph_sage_aggr,
            )
            gae_graph_sage_model.to(device)
            input_dataset.data.to(device)
            gae_graph_sage_embeddings = train_gae(
                data=input_dataset.data,
                encoder=gae_graph_sage_model,
                device=device,
                epochs=arguments.epochs,
            )
            # if we want to compress the embeddings
            if arguments.dim_reduction:
                gae_graph_sage_embeddings = dimension_reduction(
                    arguments,
                    dataset,
                    embedding_df=gae_graph_sage_embeddings,
                    label_dict=label_dict,
                    report_dir=report_dir
                    / arguments.model
                    / arguments.gae_encoder
                    / "DR",
                )
            tsne_title = f"TSNE of {arguments.model.upper()}-{arguments.gae_encoder.upper()}-{arguments.gae_feature.upper()}-{emb_dim}"  # noqa
            plot_tsne(
                emb_df=gae_graph_sage_embeddings,
                title=tsne_title,
                filename=report_dir
                / arguments.model
                / arguments.gae_encoder
                / arguments.gae_feature
                / f"tsne-{arguments.model}-{arguments.gae_encoder}-{arguments.gae_feature}-dim-{emb_dim}.png",
                label_int_2_str=label_dict,
            )
            # train classification
            gae_graph_sage_ml_model = GraphMLTrain(
                gae_graph_sage_embeddings,
                y_field_name="y",
                svm_best_param=ml_default_settings.pretrain_svm_best_param,
                knn_best_param=ml_default_settings.pretrain_knn_best_param,
                rf_best_param=ml_default_settings.pretrain_rf_best_param,
                oversampling=arguments.oversampling,
                plot_it=arguments.plot_it,
                risk_labels=label_list,
            )
            gae_graph_sage_ml_models = gae_graph_sage_ml_model.train(
                grid_search=arguments.grid_search
            )
            gae_graph_sage_ml_model_metrics = (
                gae_graph_sage_ml_model.plt_3_confusion_matrix(
                    gae_graph_sage_ml_models,
                    f"{arguments.model.upper()} Confusion Matrix",
                    plot_it=arguments.plot_it,
                    balanced_ac=arguments.balanced_ac,
                )
            )
            gae_graph_sage_unsupervised_performance[
                emb_dim
            ] = gae_graph_sage_ml_model_metrics
        for metric_name in [
            "accuracy_score",
            "macro_precision",
            "macro_recall",
            "macro_f_beta_score",
            "micro_precision",
            "micro_recall",
            "micro_f_beta_score",
            "roc_auc",
        ]:
            plot_metrics(
                metrics=gae_graph_sage_unsupervised_performance,
                title="GAE GraphSAGE Model Performance vs Dimensionality",
                x_title="Dimensionality",
                y_title=metric_name.upper(),
                metric_name=metric_name,
                filename=report_dir
                / arguments.model
                / arguments.gae_encoder
                / arguments.gae_feature
                / f"{arguments.model}-{arguments.gae_encoder}-{arguments.gae_feature}-dim-{metric_name}.png",
            )
        to_json_file(
            gae_graph_sage_unsupervised_performance,
            report_dir
            / arguments.model
            / arguments.gae_encoder
            / arguments.gae_feature
            / f"{arguments.model}-{arguments.gae_encoder}-{arguments.gae_feature}-dim-metrics.json",
        )
        return gae_graph_sage_unsupervised_performance


def handle_graph_ag_model(
    arguments: argparse.Namespace,
    ml_default_settings,
    dataset,
    # dataset_with_centrality,
    report_dir,
    label_list,
    label_dict: Optional[dict] = None,
    device: torch.device = torch.device("cpu"),
):
    logger.info("Model: Graph Analysis")
    # add the dataset.data.x to the graph_ag_embeddings
    # dataset.data.x to dataframe
    feature_df = pd.DataFrame(dataset.data.x.detach().cpu().numpy())
    feature_df["y"] = dataset.data.y.detach().cpu().numpy()
    plot_tsne(
        emb_df=feature_df,
        title="Graph Feature TSNE",
        filename=report_dir / arguments.model / f"{arguments.model}-feature.png",
    )
    # drop y from feature_df
    feature_df = feature_df.drop(columns=["y"])

    graph_ag_performance = {}
    if arguments.start_dim > arguments.end_dim:
        raise ValueError("Start dimension must be less than end dimension")
    for dim in range(arguments.start_dim, arguments.end_dim):
        graph_ag_model = GraphAGEmb(
            in_channels=dataset.data.num_features,
            hidden_channels=[32],
            out_channels=dim,
        )
        graph_ag_model.to(device)
        dataset.data.to(device)
        graph_ag_embeddings = graph_ag_model.fit(dataset.data, epochs=arguments.epochs)

        plot_tsne(
            emb_df=graph_ag_embeddings,
            title="Graph AG Model TSNE",
            filename=report_dir / arguments.model / f"{arguments.model}-{dim}.png",
            label_int_2_str=label_dict,
        )
        # concat with feature_df
        graph_ag_embeddings = pd.concat([feature_df, graph_ag_embeddings], axis=1)
        plot_tsne(
            emb_df=graph_ag_embeddings,
            title="Graph AG Model with feature TSNE",
            filename=report_dir
            / arguments.model
            / f"{arguments.model}-feature-{dim}.png",
            label_int_2_str=label_dict,
        )
        graph_age_ml_model = GraphMLTrain(
            graph_ag_embeddings,
            y_field_name="y",
            svm_best_param=ml_default_settings.pretrain_svm_best_param,
            knn_best_param=ml_default_settings.pretrain_knn_best_param,
            rf_best_param=ml_default_settings.pretrain_rf_best_param,
            oversampling=arguments.oversampling,
            plot_it=arguments.plot_it,
            risk_labels=label_list,
        )
        graph_age_ml_models = graph_age_ml_model.train(
            grid_search=arguments.grid_search
        )
        graph_age_ml_model_metrics = graph_age_ml_model.plt_3_confusion_matrix(
            graph_age_ml_models,
            f"{arguments.model.upper()} Confusion Matrix",
            plot_it=arguments.plot_it,
            balanced_ac=arguments.balanced_ac,
        )
        # show the test performance for each category
        graph_ag_performance[dim] = graph_age_ml_model_metrics
    for metric_name in [
        "accuracy_score",
        "macro_precision",
        "macro_recall",
        "macro_f_beta_score",
        "micro_precision",
        "micro_recall",
        "micro_f_beta_score",
        "roc_auc",
    ]:
        plot_metrics(
            metrics=graph_ag_performance,
            title="Graph AG Model Performance vs Dimensionality",
            x_title="Dimensionality",
            y_title=metric_name.upper(),
            metric_name=metric_name,
            filename=report_dir
            / arguments.model
            / f"{arguments.model}-dim-{metric_name}.png",
        )
    return graph_ag_performance
