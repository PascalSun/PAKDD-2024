"""
Read from all the datasets report folder, gather the best metrics for each dataset
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.constants import REPORT_DIR
from src.utils.logger import get_logger

DATASETS = [
    "Mitcham",
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
    "Flickr",
    "Yelp",
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
    "Reddit",
    "AMAZON_PRODUCTS",
]

MODELS = [
    "node2vec",
    "gae",
    "gcn",
    "graph_sage",
    "random_input",
]

SUMMARY_MODEL_COLUMN_ORDER = [
    "dataset_name",
    "model",
    "sub_model",
    "svm_accuracy_score",
    "knn_accuracy_score",
    "rf_accuracy_score",
    "svm_macro_f_beta_score",
    "knn_macro_f_beta_score",
    "rf_macro_f_beta_score",
    "svm_macro_precision",
    "knn_macro_precision",
    "rf_macro_precision",
    "svm_macro_recall",
    "knn_macro_recall",
    "rf_macro_recall",
    "svm_micro_f_beta_score",
    "knn_micro_f_beta_score",
    "rf_micro_f_beta_score",
    "svm_micro_precision",
    "knn_micro_precision",
    "rf_micro_precision",
    "svm_micro_recall",
    "knn_micro_recall",
    "rf_micro_recall",
    "svm_roc_auc",
    "knn_roc_auc",
    "rf_roc_auc",
    "direct_task_accuracy_score",
]


class IIDMetricsEvaluation:
    def __init__(self):
        self.logger = get_logger("IIDMetricsEvaluation")
        self.metric_names = [
            "accuracy_score",
            "macro_f_beta_score",
            "macro_precision",
            "macro_recall",
            "micro_f_beta_score",
            "micro_precision",
            "micro_recall",
            "roc_auc",
        ]
        similarity_dir = REPORT_DIR / "iid" / "summary" / "similarity_matrix"
        similarity_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def drop_duplicate(filename: Path):
        df = pd.read_csv(filename)
        df.drop_duplicates(inplace=True)
        # also round the number to 4 decimal places
        df = df.round(3)
        df.to_csv(filename, index=False)

    def gather_metrics(self, dataset_name: str):
        metrics = []
        report_dir = REPORT_DIR / dataset_name
        self.logger.info(f"dataset_name: {dataset_name}")
        # -------------------------------------------------------
        # get the centrality and feature_1433 json in root folder
        feature_1433_json = report_dir / "feature_1433_metrics.json"
        feature_centrality_json = report_dir / "feature_centrality_metrics.json"

        feature_1433_metric = {
            "dataset_name": dataset_name,
            "model": "feature_1433",
            "sub_model": "feature_1433",
        }
        feature_centrality_metric = {
            "dataset_name": dataset_name,
            "model": "feature_centrality",
            "sub_model": "feature_centrality",
        }
        if feature_1433_json.exists():
            feature_1433_metric = self.get_best_metrics(
                feature_1433_json, feature_1433_metric, is_json=True
            )
            metrics.append(feature_1433_metric)
        if feature_centrality_json.exists():
            feature_centrality_metric = self.get_best_metrics(
                feature_centrality_json, feature_centrality_metric, is_json=True
            )
            metrics.append(feature_centrality_metric)

        # ------------------------------------------------------------------

        # loop the folders in models, get the json file
        for model in MODELS:
            model_report_dir = report_dir / model
            if not model_report_dir.exists():
                continue

            # recursively find the json file, and get the best metrics within each json file
            json_files = list(model_report_dir.glob("**/*.json"))
            self.logger.info(f"json_files: {model}({len(json_files)})")
            if len(json_files) == 0:
                continue
            # get the best metrics
            for json_file in json_files:
                model_metric = {
                    "dataset_name": dataset_name,
                }
                # get the name difference between the json file and the model report dir
                self.logger.info(f"json_file: {json_file}")
                folder_name = (
                    json_file.as_posix()
                    .replace(f"{model_report_dir.as_posix()}/", "")
                    .rsplit("/", 1)[0]
                    .replace("/", "-")
                )
                model_metric["model"] = model
                model_metric["sub_model"] = folder_name
                self.logger.info(f"model_metric: {model_metric}")

                self.logger.info(f"folder_name: {folder_name}")
                if folder_name.endswith(".json") and "random" not in folder_name:
                    """
                    If it is json, there are two cases:
                    - root folder, feature_1433 and centrality, it will have SVM/KNN/RF, which we have handled above
                    - sub folder, feature_1433 and centrality, it will be the performance with direct task
                    """
                    json_metric = json.load(json_file.open())
                    # self.logger.info(f"model_metric: {json_metric}")
                    for metric_name in self.metric_names:
                        if metric_name != "accuracy_score":
                            continue
                        model_metric[f"direct_task_{metric_name}"] = json_metric[
                            metric_name
                        ]
                else:
                    model_metric = self.get_best_metrics(Path(json_file), model_metric)

                metrics.append(model_metric)
        # ------------------------------------------------------------------
        # output should be in the following format to a csv file
        """
        headers: dataset_name, model, svm_accuracy_score, knn_accuracy_score, rf_accuracy_score, ...
        """
        self.logger.info(f"metrics: {metrics}")
        for metric in metrics:
            for metric_key in SUMMARY_MODEL_COLUMN_ORDER:
                if metric_key not in metric:
                    metric[metric_key] = None
        # write to csv, summary folder
        summary_file = REPORT_DIR / "iid" / "summary" / "summary_models.csv"
        # convert the list to dataframe
        if len(metrics) == 0:
            return
        df = pd.DataFrame(metrics)

        df = df[SUMMARY_MODEL_COLUMN_ORDER]

        # if the file exists, append to the file
        if summary_file.exists():
            df.to_csv(summary_file, mode="a", header=False, index=False)
        else:
            # write to csv
            df.to_csv(summary_file, index=False)

        self.drop_duplicate(summary_file)

    def get_best_metrics(  # noqa
            self, file: Path, model_metric: dict, is_json: bool = False
    ):
        """
        For each metric

        - accuracy_score
        - f1_score
        - precision_score
        - recall_score
        - roc_auc_score

        get the best metrics
        Parameters
        ----------
        file
        model_metric
        is_json

        Returns
        -------

        """
        for metric_key in self.metric_names:
            metrics = json.load(file.open())
            best_metrics = {
                "SVM": 0,
                "KNN": 0,
                "RF": 0,
            }
            if is_json:
                for item in metrics:
                    if item["name"] == "SVM":
                        if (
                                item[metric_key] is not None
                                and item[metric_key] > best_metrics["SVM"]
                        ):
                            best_metrics["SVM"] = item[metric_key]
                    elif item["name"] == "KNN":
                        if (
                                item[metric_key] is not None
                                and item[metric_key] > best_metrics["KNN"]
                        ):
                            best_metrics["KNN"] = item[metric_key]
                    elif item["name"] == "RF":
                        if (
                                item[metric_key] is not None
                                and item[metric_key] > best_metrics["RF"]
                        ):
                            best_metrics["RF"] = item[metric_key]
            else:
                for key, value in metrics.items():
                    if type(value) is not list:
                        continue
                    for item in value:
                        if item[metric_key] is not None and item["name"] == "SVM":
                            if item[metric_key] > best_metrics["SVM"]:
                                best_metrics["SVM"] = item[metric_key]
                        elif item[metric_key] is not None and item["name"] == "KNN":
                            if item[metric_key] > best_metrics["KNN"]:
                                best_metrics["KNN"] = item[metric_key]
                        elif item[metric_key] is not None and item["name"] == "RF":
                            if item[metric_key] > best_metrics["RF"]:
                                best_metrics["RF"] = item[metric_key]
            self.logger.info(f"best_metrics: {best_metrics}")
            model_metric[f"svm_{metric_key}"] = best_metrics["SVM"]
            model_metric[f"knn_{metric_key}"] = best_metrics["KNN"]
            model_metric[f"rf_{metric_key}"] = best_metrics["RF"]
        return model_metric

    @staticmethod
    def rank_models(logger=None):
        """
        Rank the models based on the metrics
        return will be like

        dataset, rank_1_model, rank_1_model_score, rank_2_model, rank_2_model_score, rank_3_model, rank_3_model_score
        Returns
        -------

        """
        # read the csv file
        if not logger:
            logger = get_logger("IIDMetricsEvaluation")
        (REPORT_DIR / "iid" / "summary" / "rank").mkdir(parents=True, exist_ok=True)
        summary_file = REPORT_DIR / "iid" / "summary" / "summary_models.csv"
        df = pd.read_csv(summary_file)
        # for each dataset, and each metric, rank the model

        for metric_name in [
            "accuracy_score",
            "roc_auc",
            "macro_f_beta_score",
            "macro_precision",
            "macro_recall",
            "micro_f_beta_score",
            "micro_precision",
            "micro_recall",
        ]:
            dataset_metrics = []
            for dataset_name in DATASETS:
                models_metric = df[df["dataset_name"] == dataset_name]
                # check accuracy score
                metrics_list = []
                for _, row in models_metric.iterrows():
                    row_dict = row.to_dict()
                    for key in row_dict:
                        if metric_name in key:
                            # print(row_dict[key])
                            # print(key)
                            if pd.isna(row_dict[key]):
                                continue

                            metrics_list.append(
                                (
                                    f"{row_dict['model']}_{row_dict['sub_model']}_{key}",
                                    row_dict[key],
                                )
                            )
                metrics_list.sort(key=lambda x: x[1], reverse=True)
                # map the list to dict, and format will be rank_1_model_name, rank_1_model_score,
                # rank_2_model_name, rank_2_model_score
                metrics_dict = {}
                # logger.info(f"metrics_list length: {len(metrics_list)}")
                if len(metrics_list) <= 30:
                    logger.critical(
                        f"dataset {dataset} has less than 30 models to evaluate"
                    )
                for index, item in enumerate(metrics_list):
                    metrics_dict[f"rank_{index + 1}_model_name"] = item[0]
                    metrics_dict[f"rank_{index + 1}_model_score"] = item[1]

                metrics_dict["dataset"] = dataset_name

                dataset_metrics.append(metrics_dict)

            # output the rank to csv
            rank_file = (
                    REPORT_DIR
                    / "iid"
                    / "summary"
                    / "rank"
                    / f"rank_{metric_name}_models.csv"
            )
            rank_df = pd.DataFrame(dataset_metrics)
            # put the dataset column to the first column
            cols = rank_df.columns.tolist()
            # get the index of dataset
            dataset_index = cols.index("dataset")
            cols = (
                    [cols[dataset_index]] + cols[:dataset_index] + cols[dataset_index + 1:]
            )
            rank_df = rank_df[cols]

            rank_df.to_csv(rank_file, index=False)

    def plot_performance_distribution(self, logger=None):  # noqa
        """
        plot the performance distribution for each dataset,
        and the x-axis will be the same sequence for all dataset
        -------

        """
        # read the csv file
        if not logger:
            logger = get_logger("IIDMetricsEvaluation")

        # for each dataset, and each metric, rank the model

        for metric_name in [
            "accuracy_score",
            "roc_auc",
            "macro_f_beta_score",
            "macro_precision",
            "macro_recall",
            "micro_f_beta_score",
            "micro_precision",
            "micro_recall",
        ]:
            # output the rank to csv
            rank_file = (
                    REPORT_DIR
                    / "iid"
                    / "summary"
                    / "rank"
                    / f"rank_{metric_name}_models.csv"
            )
            rank_df = pd.read_csv(rank_file)
            logger.debug(rank_df.columns.tolist())
            model_names = []
            for column in rank_df.columns.tolist():
                if "model_name" in column:
                    model_names += list(set(rank_df[column].tolist()))
                    # remove pd.nan from the list
                    model_names = [item for item in model_names if type(item) is str]
            # remove the model_name start with graph_sage_graph_sage_metrics.
            model_names = [
                item
                for item in model_names
                if "graph_sage_graph_sage_metrics." not in item
            ]
            model_names = list(set(model_names))
            model_names.sort()
            logger.info(model_names)
            logger.debug(len(model_names))
            x_axis_sequence = model_names

            # for each dataset, plot the distribution, rank_model_score is the y, rank_model_name is the x
            # we will first need to construct the dataset
            # iterate the dataset for rows
            finished_dataset = []
            metric_vectors = {}
            for _, row in rank_df.iterrows():
                performance_data = {"dataset": row.dataset}
                row = row.to_dict()
                output_dir = REPORT_DIR / "iid" / "summary" / metric_name
                output_dir.mkdir(parents=True, exist_ok=True)

                for column in rank_df.columns.tolist():
                    if "score" in column:
                        rank = column.split("_")[1]
                        rank_model_name_column = f"rank_{rank}_model_name"
                        if row[rank_model_name_column] not in x_axis_sequence:
                            continue
                        if pd.isna(row[column]):
                            continue
                        else:
                            performance_data[row[rank_model_name_column]] = row[column]
                logger.debug(performance_data)
                for model_name in x_axis_sequence:
                    if model_name not in performance_data:
                        performance_data[model_name] = 0

                # temporarily remove other node2vec models, only keep node2vec_embedding_dim one
                # logic will be: loop the dict, if node2vec in the key and node2vec_embedding_dim not in the key
                # then remove the key/value pair
                for key in list(performance_data.keys()):
                    if "node2vec" in key and "node2vec_embedding_dim" not in key:
                        del performance_data[key]

                # plot the performance distribution, x-axis will be the model name, y-axis will be the score
                # also x-axis will be the same for all dataset, use the x_axis_sequence, use plotly to plot
                sorted_keys = [
                    key for key in x_axis_sequence if key in performance_data
                ]
                sorted_values = [performance_data[key] for key in sorted_keys]
                # for each sorted_key, remove metric_name from the string
                sorted_keys = [sorted_key.replace(f"_{metric_name}", '') for sorted_key in sorted_keys]
                # for key start with feature_1433, feature_centrality, random_input, remove the first feature_1433, feature_centrality, random_input in the string
                updated_sorted_keys = []
                for sorted_key in sorted_keys:
                    if sorted_key.startswith("feature_1433"):
                        new_sorted_key = sorted_key.replace("feature_1433_", "", 1)
                    elif sorted_key.startswith("feature_centrality"):
                        new_sorted_key = sorted_key.replace("feature_centrality_", "", 1)
                    elif sorted_key.startswith("random_input"):
                        new_sorted_key = sorted_key.replace("random_input_", "", 1)
                    elif "direct_task" in sorted_key:
                        if ("gcn" in sorted_key) and ("centrality" in sorted_key):
                            new_sorted_key = "gcn_centrality_direct_task"
                        elif ("gcn" in sorted_key) and ("feature" in sorted_key):
                            new_sorted_key = "gcn_feature_direct_task"
                        elif ("graph_sage" in sorted_key) and ("centrality" in sorted_key):
                            new_sorted_key = "graph_sage_centrality_direct_task"
                        elif ("graph_sage" in sorted_key) and ("feature" in sorted_key):
                            new_sorted_key = "graph_sage_feature_direct_task"
                        else:
                            print(sorted_key)
                            new_sorted_key = None
                    else:
                        new_sorted_key = sorted_key
                    updated_sorted_keys.append(new_sorted_key)
                # replace - to _, replace feature_1433 to feature
                updated_sorted_keys = [sorted_key.replace("-", "_") for sorted_key in updated_sorted_keys]
                updated_sorted_keys = [sorted_key.replace("feature_1433_", "feature_") for sorted_key in
                                       updated_sorted_keys]
                # replace number 1 in the key
                updated_sorted_keys = [sorted_key.replace("1", "") for sorted_key in updated_sorted_keys]
                updated_sorted_keys = [
                    "feature_knn",
                    "feature_rf",
                    "feature_svm",
                    "centrality_knn",
                    "centrality_rf",
                    "centrality_svm",
                    "gae_gcn_centrality_knn",
                    "gae_gcn_centrality_rf",
                    "gae_gcn_centrality_svm",
                    "gae_gcn_feature_knn",
                    "gae_gcn_feature_rf",
                    "gae_gcn_feature_svm",
                    "gae_graph_sage_centrality_knn",
                    "gae_graph_sage_centrality_rf",
                    "gae_graph_sage_centrality_svm",
                    "gae_graph_sage_feature_knn",
                    "gae_graph_sage_feature_rf",
                    "gae_graph_sage_feature_svm",
                    "gcn_centrality_knn",
                    "gcn_centrality_rf",
                    "gcn_centrality_svm",
                    "gcn_feature_knn",
                    "gcn_feature_rf",
                    "gcn_feature_svm",
                    "gcn_centrality_direct_task",
                    "gcn_feature_direct_task",
                    "graph_sage_centrality_knn",
                    "graph_sage_centrality_rf",
                    "graph_sage_centrality_svm",
                    "graph_sage_feature_knn",
                    "graph_sage_feature_rf",
                    "graph_sage_feature_svm",
                    "graph_sage_centrality_direct_task",
                    "graph_sage_feature_direct_task",
                    "node2vec_knn",
                    "node2vec_rf",
                    "node2vec_svm",
                    "random_knn",
                    "random_rf",
                    "random_svm"
                ]

                def get_symbol(sort_key_name):
                    if sort_key_name.endswith('knn'):
                        return 'circle'
                    elif sort_key_name.endswith('rf'):
                        return 'square'
                    elif sort_key_name.endswith('svm'):
                        return 'diamond'
                    elif sort_key_name.endswith('direct_task'):
                        return 'cross'
                    else:
                        return 'circle'  # default shape

                symbols = [get_symbol(key) for key in updated_sorted_keys]
                # first 6 will be similar colors, but two different color
                color_list = [
                    # First 6 (3 + 3)
                    '#FF0000', '#FF0000', '#FF0000', '#FF3300', '#FF3300', '#FF3300',

                    # 7-18 (6 + 6, with 3 + 3 within each 6)
                    '#00FF00', '#00FF00', '#00FF00', '#33FF00', '#33FF00', '#33FF00',
                    '#00FF33', '#00FF33', '#00FF33', '#33FF33', '#33FF33', '#33FF33',

                    # Next 6 + 2 (same pattern)
                    '#0000FF', '#0000FF', '#0000FF', '#3300FF', '#3300FF', '#3300FF',
                    '#0033FF', '#0033FF',

                    # Next 6 + 2 (same pattern)
                    '#FF00FF', '#FF00FF', '#FF00FF', '#FF33FF', '#FF33FF', '#FF33FF',
                    '#FF00CC', '#FF00CC',

                    # Last 6 (3 + 3)
                    '#00FFFF', '#00FFFF', '#00FFFF', 'black', 'black', 'black'
                ]

                fig = go.Figure(
                    data=go.Scatter(
                        x=updated_sorted_keys,
                        y=sorted_values,
                        mode="markers+text",
                        text=sorted_values,
                        textposition="top center",
                        marker=dict(
                            symbol=symbols,
                            size=12,
                            color=color_list,
                            opacity=0.7,
                            line=dict(
                                color='DarkSlateGrey',
                                width=0.5
                            )
                        ),
                        textfont=dict(
                            size=14,
                            color='black'
                        )
                    )
                )
                dataset_name = row["dataset"]
                if "auther" in dataset_name:
                    dataset_name = dataset_name.replace("auther", "author")
                fig.update_layout(
                    title=dict(
                        text=f"Scatter Plot of {' '.join(metric_name.upper().split('_'))}, Dataset: {dataset_name}",
                        font=dict(size=24)),
                    xaxis_title="Model",
                    xaxis_tickangle=-45,
                    yaxis_title=' '.join(metric_name.upper().split('_')),
                    width=1500,
                    height=800,
                    yaxis=dict(range=[0, 1.05], gridcolor='lightgray', showgrid=True),
                    font=dict(size=18),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        gridcolor='lightgray',
                        showgrid=True,
                    )
                )
                fig.write_image(
                    REPORT_DIR
                    / "iid"
                    / "summary"
                    / metric_name
                    / f"{row['dataset']}_{metric_name}.pdf"
                )
                # if all values > 0, then it is a finished dataset
                if all(value > 0 for value in sorted_values):
                    if row["dataset"] != "Mitcham":
                        finished_dataset.append(row["dataset"])

                        # ----------
                        # treat the sorted_values as vectors
                        metric_vectors[row["dataset"]] = sorted_values
                        # ----------

            if metric_name == "accuracy_score" or metric_name == 'macro_f_beta_score':
                finished_dataset = list(set(finished_dataset))
                finished_dataset.sort()
                # loop to log all the finished dataset
                for fd in finished_dataset:
                    logger.info(f"finished dataset: {fd}")
                logger.info(f"finished dataset length: {len(finished_dataset)}")
                logger.info(finished_dataset)

                # also plot the similarity matrix for the network metrics
                # read the summary.csv file, and get all the related metrics for datasets within finished_dataset
                network_summary_df = pd.read_csv(
                    REPORT_DIR
                    / "iid"
                    / "summary"
                    / "summary_network_metrics_min_max.csv"
                )

                # apply log to the num_edges
                network_summary_df["ln_num_edges"] = network_summary_df[
                    "num_edges"
                ].apply(lambda x: math.log(x))
                network_summary_df = network_summary_df[
                    network_summary_df["dataset"].isin(finished_dataset)
                ]
                self.logger.info(f"network_summary_df: {network_summary_df.shape}")

                # aim for different subset of columns combination, and plot the similarity matrix
                # row_columns_list = [
                #     "num_edges",
                #     "k",
                #     "ln_n",
                #     "std_k",
                #     "k_min",
                #     "k_max",
                #     "d_max_random",
                #     "k2",
                #     "gamma_degree",
                #     "diameter",
                #     "average_clustering_coefficient",
                #     "average_shortest_path_length",
                #     "num_classes",
                #     "num_features",
                #     "num_nodes",
                #     "transitivity",
                #     # p_k_max
                #     # ks_statistic
                # ]
                row_columns_list = [
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
                if metric_name == 'accuracy_score':
                    base_similarity = self.plot_similarity_matrix(
                        metric_vectors,
                        REPORT_DIR
                        / "iid"
                        / "summary"
                        / "similarity_matrix"
                        / f"similarity_matrix_{metric_name}.pdf",
                        title=f"Similarity Matrix for {metric_name}",
                    )

                    metric_vectors_df = pd.DataFrame(metric_vectors)
                    # transpose the metric_vectors_df
                    metric_vectors_df = metric_vectors_df.transpose()
                    metric_vectors_df.to_csv(REPORT_DIR / "iid" / "summary" / "accuracy_score_metrics_vector.csv")

                if metric_name == 'macro_f_beta_score':
                    f1_similarity = self.plot_similarity_matrix(
                        metric_vectors,
                        REPORT_DIR
                        / "iid"
                        / "summary"
                        / "similarity_matrix"
                        / f"similarity_matrix_{metric_name}.pdf",
                        title=f"Similarity Matrix for {metric_name}",
                    )
                    # f1 score SSIM with base
                    self.logger.critical("ffffffffffffffffffffffffffffffffffffffffff")
                    self.logger.critical(self.ssim(base_similarity, f1_similarity))
                # get metric vectors to pandas and to csv

                # Generate all combinations of the elements ranging from size 2 to the length of the list
                row_columns_combinations = []
                for r in range(2, len(row_columns_list) + 1):
                    row_columns_combinations.extend(
                        itertools.combinations(row_columns_list, r)
                    )
                #
                row_columns_combinations = [
                    ('ln_num_edges', 'ln_num_classes', 'ln_average_shortest_path_length'),
                    ("ln_num_classes", "ln_n", "ln_k_max"),
                    ("ln_num_classes", "ln_n", "ln_average_shortest_path_length"),
                    (
                        "ln_num_classes",
                        "ln_n",
                        "ln_k_max",
                        "ln_average_shortest_path_length",
                    ),
                    ('ln_num_edges', 'ln_num_classes', 'ln_k2')
                ]
                # row_columns_combinations = [('ln_num_edges', 'num_features', 'ln_num_classes', 'ln_n', 'k', 'std_k',
                #                              'ln_k2', 'k_min', 'ln_k_max',
                #                              'ln_d_max_random', 'gamma_degree', 'ln_average_shortest_path_length',
                #                              'ln_diameter',
                #                              'average_clustering_coefficient', 'transitivity')]
                # need a list of list to store the combinations results
                # and then sort the list by the least value of the diff
                diff_results = []
                self.logger.info(
                    f"row_columns_combinations: {len(row_columns_combinations)}"
                )
                for row_columns in row_columns_combinations:
                    self.logger.critical(row_columns)
                    network_metric_vectors = {}
                    for _, row in network_summary_df.iterrows():
                        network_metric_vectors[row["dataset"]] = row[list(row_columns)]
                    # self.logger.info(f"metric_vectors: {network_metric_vectors}")
                    compare_similarity = self.plot_similarity_matrix(
                        network_metric_vectors,
                        REPORT_DIR
                        / "iid"
                        / "summary"
                        / "similarity_matrix"
                        / f"similarity_matrix_network_{'_'.join(row_columns)}.pdf",
                        title=f"Similarity Matrix for Network Metrics ({row_columns})",
                    )
                    # there are several ways to compare the similarity between matrices
                    # 1. total difference
                    # 2. relative error for the Frobenius norm
                    # 3. Frobenius distance
                    # 4. cosine similarity
                    # 5. spectral distance
                    # 6. matrix correlation

                    total_diff_similarity = self.total_diff_similarity(
                        base_similarity, compare_similarity
                    )
                    relative_error = self.relative_error(
                        base_similarity, compare_similarity
                    )
                    fro_similarity = self.frobenius_distance(
                        base_similarity, compare_similarity
                    )
                    cosine_similarity_value = self.cosine_similarity(
                        base_similarity, compare_similarity
                    )
                    spectral_distance = self.spectral_distance(
                        base_similarity, compare_similarity
                    )
                    matrix_correlation = self.matrix_correlation(
                        base_similarity, compare_similarity
                    )
                    dot_product = self.dot_product(base_similarity, compare_similarity)
                    average_row_dot_product = self.element_wise_dot_product(
                        base_similarity, compare_similarity
                    )
                    row_by_row_cosine_similarity = self.row_by_row_cosine_similarity(
                        base_similarity, compare_similarity
                    )
                    ssim_similarity = self.ssim(base_similarity, compare_similarity)
                    self.logger.critical(f"ssim_similarity: {ssim_similarity}")
                    diff_results.append(
                        (
                            row_columns,
                            total_diff_similarity,
                            relative_error,
                            fro_similarity,
                            cosine_similarity_value,
                            spectral_distance,
                            matrix_correlation,
                            dot_product,
                            average_row_dot_product,
                            row_by_row_cosine_similarity,
                            ssim_similarity,
                        )
                    )

                    self.logger.critical(
                        "--------------------------------------------------------"
                    )
                # diff_results.sort(key=lambda x: x[1])
                # for diff_result in diff_results:
                #     self.logger.critical(diff_result)
                # # convert to pandas and then save to csv
                diff_results_df = pd.DataFrame(diff_results)
                # give df the column names
                diff_results_df.columns = [
                    "row_columns",
                    "total_diff_similarity",
                    "relative_error",
                    "fro_similarity",
                    "cosine_similarity_value",
                    "spectral_distance",
                    "matrix_correlation",
                    "dot_product",
                    "average_row_dot_product",
                    "row_by_row_cosine_similarity",
                    "ssim_similarity",
                ]
                # sort by the row_by_row_cosine_similarity descending
                diff_results_df.sort_values(
                    by=["row_by_row_cosine_similarity"], ascending=False, inplace=True
                )

                diff_results_df.to_csv(
                    REPORT_DIR
                    / "iid"
                    / "summary"
                    / "similarity_matrix_network_diff_results_1114.csv",
                    index=False,
                )

            # ----------------------------------------------------------
            # we will want to calculate similarity between the vectors, and plot the similarity matrix
            self.plot_similarity_matrix(
                metric_vectors,
                REPORT_DIR
                / "iid"
                / "summary"
                / "similarity_matrix"
                / f"similarity_matrix_{metric_name}.pdf",
                title=f"Similarity Matrix for {metric_name}",
            )

    @staticmethod
    def plot_similarity_matrix(metrics_vectors: dict, output_image: Path, title: str):
        # calculate the similarity matrix
        # we will use cosine similarity
        names = list(metrics_vectors.keys())
        names.sort()
        vectors = [metrics_vectors[name] for name in names]

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(vectors)
        A = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())

        # Create a structured result
        # result = {}
        # for i, name1 in enumerate(names):
        #     for j, name2 in enumerate(names):
        #         result[(name1, name2)] = similarity_matrix[i][j]

        # plot the similarity matrix, get it to be a square matrix
        fig = go.Figure(data=go.Heatmap(z=A, x=names, y=names))
        # get the x and y label to be larger
        fig.update_layout(
            xaxis=dict(
                tickfont=dict(
                    size=20,
                )
            ),
            yaxis=dict(
                tickfont=dict(
                    size=20,
                ),
                autorange="reversed",  # Set y-axis to be reversed
            ),
        )
        fig.update_layout(width=2000, height=2000)
        # add title
        fig.update_layout(title=title)
        # get the title to be larger
        fig.update_layout(title_font_size=30)
        fig.write_image(output_image)

        return similarity_matrix

    @staticmethod
    def total_diff_similarity(A, B):
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())
        diff = np.abs(A - B)

        # Sum up the absolute differences
        total_difference = np.sum(diff)
        return total_difference

    @staticmethod
    def relative_error(A, B):
        """
        Compute the relative error between two matrices A and B using the Frobenius norm.

        Args:
        - A (np.ndarray): Reference matrix
        - B (np.ndarray): Matrix to compare

        Returns:
        - float: Relative error
        """
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())
        # Compute the Frobenius norm of the difference
        diff_norm = np.linalg.norm(A - B, "fro")

        # Compute the Frobenius norm of the reference matrix A
        ref_norm = np.linalg.norm(A, "fro")

        # Handle the case where the reference matrix is the zero matrix
        if ref_norm == 0:
            raise ValueError(
                "The reference matrix is the zero matrix, so relative error is undefined."
            )

        return diff_norm / ref_norm

    @staticmethod
    def frobenius_distance(A, B):
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())
        return np.linalg.norm(A - B, "fro")

    @staticmethod
    def cosine_similarity(A, B):
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())
        A_vec = A.ravel()
        B_vec = B.ravel()
        dot_product = np.dot(A_vec, B_vec)
        norm_A = np.linalg.norm(A_vec)
        norm_B = np.linalg.norm(B_vec)
        return dot_product / (norm_A * norm_B)

    @staticmethod
    def dot_product(A, B):
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())
        dot_p = np.dot(A, B)
        return np.linalg.norm(dot_p, 2)

    @staticmethod
    def element_wise_dot_product(A, B):
        """
        Do dot product for each row of A and B
        and then normalize the result by the norm of A and B
        then average the result of all rows
        Parameters
        ----------
        A
        B

        Returns
        -------

        """
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())
        # Compute the dot product for each row
        dot_products = np.sum(A * B, axis=1)

        # Normalize the result by the norm of A and B for each row
        norms_A = np.linalg.norm(A, axis=1)
        norms_B = np.linalg.norm(B, axis=1)

        normalized_products = dot_products / (norms_A * norms_B)

        # Average the result of all rows
        average_result = np.mean(normalized_products)

        return average_result

    @staticmethod
    def row_by_row_cosine_similarity(A, B):
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())
        num_rows = A.shape[0]
        similarities = np.zeros(num_rows)

        for i in range(num_rows):
            similarities[i] = cosine_similarity(
                A[i].reshape(1, -1), B[i].reshape(1, -1)
            )

        return similarities.mean()

    @staticmethod
    def spectral_distance(A, B):
        return np.linalg.norm(A - B, 2)

    @staticmethod
    def matrix_correlation(A, B):
        A_vec = A.ravel()
        B_vec = B.ravel()
        return np.corrcoef(A_vec, B_vec)[0, 1]

    @staticmethod
    def ssim(A, B):
        matrix1 = (A - A.min()) / (A.max() - A.min())
        matrix2 = (B - B.min()) / (B.max() - B.min())
        return ssim(matrix1, matrix2, data_range=1.0)

    @staticmethod
    def plot_performance_spectrum():
        metrics_csv = REPORT_DIR / "iid" / "summary" / "accuracy_score_metrics_vector.csv"
        df = pd.read_csv(metrics_csv)
        import plotly.express as px

        # reset index, and give it name ["model", 0-39]

        df.columns = ["dataset"] + list(range(40))
        # sort by the dataset column
        df.sort_values(by=["dataset"], inplace=True)
        print(df)

        # Min-Max Scaling for each row
        df_scaled = df.iloc[:, 1:].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

        # Extracting labels and scaled values
        labels = df.iloc[:, 0].tolist()
        print(labels)
        values = df_scaled.values
        print(values.shape)
        updated_sorted_keys = [
            "feature_knn",
            "feature_rf",
            "feature_svm",
            "centrality_knn",
            "centrality_rf",
            "centrality_svm",
            "gae_gcn_centrality_knn",
            "gae_gcn_centrality_rf",
            "gae_gcn_centrality_svm",
            "gae_gcn_feature_knn",
            "gae_gcn_feature_rf",
            "gae_gcn_feature_svm",
            "gae_graph_sage_centrality_knn",
            "gae_graph_sage_centrality_rf",
            "gae_graph_sage_centrality_svm",
            "gae_graph_sage_feature_knn",
            "gae_graph_sage_feature_rf",
            "gae_graph_sage_feature_svm",
            "gcn_centrality_knn",
            "gcn_centrality_rf",
            "gcn_centrality_svm",
            "gcn_feature_knn",
            "gcn_feature_rf",
            "gcn_feature_svm",
            "gcn_centrality_direct_task",
            "gcn_feature_direct_task",
            "graph_sage_centrality_knn",
            "graph_sage_centrality_rf",
            "graph_sage_centrality_svm",
            "graph_sage_feature_knn",
            "graph_sage_feature_rf",
            "graph_sage_feature_svm",
            "graph_sage_centrality_direct_task",
            "graph_sage_feature_direct_task",
            "node2vec_knn",
            "node2vec_rf",
            "node2vec_svm",
            "random_knn",
            "random_rf",
            "random_svm"
        ]
        # Creating heatmap
        fig = px.imshow(values,
                        labels=dict(x="Dimension", y="Dataset", color="Scaled Value"),
                        x=updated_sorted_keys,  # Setting x to match the number of columns in 'values'
                        y=labels,
                        color_continuous_scale='viridis',  # Using the viridis color scale
                        aspect="auto"  # This will ensure the cells are stretched to fit the width
                        )

        fig.update_layout(
            title_font=dict(size=24),
            height=len(labels) * 50, width=2000)  # Setting height based on number of rows
        fig.update_layout(
            title=r'$\text{Node2Vec vs Feature Only Performance} \left( \frac{\overline{\text{node2vec}} - \overline{\text{Random Guess}}}{\overline{\text{feature only}} - \overline{\text{Random Guess}}} \right)$'
        )
        fig.update_xaxes(tickfont=dict(size=20))
        fig.update_yaxes(tickfont=dict(size=20))
        fig.write_image(
            REPORT_DIR / "iid" / "summary" / "accuracy_score_metrics_vector.pdf"
        )

    @staticmethod
    def get_node2vec_metrics():
        metric_df = pd.read_csv(REPORT_DIR / "iid" / "summary" / "accuracy_score_metrics_vector.csv")
        # first average the first 3 columns: avg_feature
        # then average the last 3 columns: avg_random
        # then average the last 4-6 columns: avg_node2vec
        # then (avg_node2vec - avg_random) / (avg_feature - avg_random)
        metric_df["avg_feature"] = metric_df.apply(lambda row: np.max(row[1:4]), axis=1)
        metric_df["avg_random"] = metric_df.apply(lambda row: np.mean(row[38:41]), axis=1)
        metric_df["avg_node2vec"] = metric_df.apply(lambda row: np.mean(row[35:38]), axis=1)

        # rename the first column to dataset
        metric_df.rename(columns={"Unnamed: 0": "dataset"}, inplace=True)
        # filter out the dataset start with TWITCH
        # metric_df = metric_df[~metric_df["dataset"].str.startswith("TWITCH")]
        metric_df["node2vec_feature_ratio"] = (metric_df["avg_node2vec"] - metric_df["avg_random"]) / (
                metric_df["avg_feature"] - metric_df["avg_random"])
        # sort by the node2vec_feature_ratio, and plot bar chart for each dataset
        metric_df.sort_values(by=["node2vec_feature_ratio"], inplace=True)
        # plot the bar chart
        fig = go.Figure(data=go.Bar(y=metric_df["node2vec_feature_ratio"], x=metric_df["dataset"]))
        # fig.update_layout(title=f"Node2Vec vs Feature Only Performance (Avg(node2vec) - Avg(Random Guess)) / (Avg(feature only) - Avg(Random Guess))", title_font=dict(size=24))
        fig.update_layout(
            title=r'$\text{Structure Only vs Feature Only Performance} \left( \frac{\overline{\text{node2vec}} - \overline{\text{Random Guess}}}{\overline{\text{Feature Only}} - \overline{\text{Random Guess}}} \right)$'
        )
        fig.update_layout(width=2000, height=800)
        fig.update_xaxes(tickfont=dict(size=20))
        fig.update_yaxes(tickfont=dict(size=20))
        fig.write_image(
            REPORT_DIR / "iid" / "summary" / "node2vec_feature_ratio.pdf"
        )
        # output a dict for me, where the key is the dataset, and the value is cross if not need graph, and circle if need graph
        need_graph_dict = {}
        for _, row in metric_df.iterrows():
            if row["node2vec_feature_ratio"] > 0.5:
                need_graph_dict[row["dataset"]] = "circle"
            else:
                need_graph_dict[row["dataset"]] = "cross"
        print(need_graph_dict)

    @staticmethod
    def get_need_graph():
        metric_df = pd.read_csv(REPORT_DIR / "iid" / "summary" / "accuracy_score_metrics_vector.csv")
        # first average the first 3 columns: avg_feature
        # get the max for the rest of the columns except the last 3 columns
        # compare the avg_feature and the max of the rest of the columns
        # if avg_feature > max of the rest of the columns, then do not need graph
        # if avg_feature < max of the rest of the columns, then need graph
        metric_df["avg_feature"] = metric_df.apply(lambda row: np.max(row[1:4]), axis=1)
        # calculate the avg for the next 3,3,3,3,3,3,3,ignore 2 column,3,3, ignore2, 3

        metric_df["max_rest"] = metric_df.apply(lambda row: np.max(row[4:38]), axis=1)

        metric_df["need_graph"] = metric_df.apply(lambda row: row["avg_feature"] < row["max_rest"], axis=1)
        # rename the first column to dataset
        metric_df.rename(columns={"Unnamed: 0": "dataset"}, inplace=True)
        # get an attribute: (avg_feature - max_rest) / avg_feature
        metric_df["feature_graph_ratio"] = (metric_df["max_rest"] - metric_df["avg_feature"]) / metric_df["avg_feature"]
        # sort by the feature_graph_ratio, and plot bar chart for each dataset
        metric_df.sort_values(by=["feature_graph_ratio"], inplace=True)
        # plot the bar chart
        fig = go.Figure(data=go.Bar(y=metric_df["feature_graph_ratio"], x=metric_df["dataset"]))

        fig.update_layout(width=2000, height=800)
        fig.update_xaxes(tickfont=dict(size=20))
        fig.update_yaxes(tickfont=dict(size=20))
        fig.write_image(
            REPORT_DIR / "iid" / "summary" / "feature_graph_ratio.pdf"
        )

        # output a dict for me, where the key is the dataset, and the value is cross if not need graph, and circle if need graph
        need_graph_dict = {}
        for _, row in metric_df.iterrows():
            if row["need_graph"]:
                need_graph_dict[row["dataset"]] = "circle"
            else:
                need_graph_dict[row["dataset"]] = "cross"
        print(need_graph_dict)


if __name__ == "__main__":
    model_summary_file = REPORT_DIR / "iid" / "summary" / "summary_models.csv"
    if model_summary_file.exists():
        model_summary_file.unlink()  # delete the file
    iid_metrics_evaluation = IIDMetricsEvaluation()
    for dataset in DATASETS:
        iid_metrics_evaluation.gather_metrics(dataset)
    # rank the models performance for each dataset
    iid_metrics_evaluation.rank_models()
    # plot the performance for different models of each dataset.
    # and gather the similarity matrix for each pair of datasets
    iid_metrics_evaluation.plot_performance_distribution()
    iid_metrics_evaluation.plot_performance_spectrum()
    iid_metrics_evaluation.get_node2vec_metrics()
    # iid_metrics_evaluation.get_need_graph()
