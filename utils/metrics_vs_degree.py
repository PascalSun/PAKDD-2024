# import pandas as pd
#
# from src.utils.constants import DATA_DIR, REPORT_DIR


class ClassificationMetricsVSGraphMetrics:
    """
    Evaluate the classification metrics vs graph metrics

    This chunk of accuracy metrics is especially important for node2vec in my opinion.

    1. class label | node_in_degree | accuracy | precision | recall | f1_score | support
    2. class label | node_out_degree | accuracy | precision | recall | f1_score | support
    3. class label | node_degree | accuracy | precision | recall | f1_score | support
    4. class label | average_in_degree | accuracy | precision | recall | f1_score | support
    5. class label | average_out_degree | accuracy | precision | recall | f1_score | support
    6. class label | average_degree | accuracy | precision | recall | f1_score | support

    """

    def __init__(
        self,
        dataset_name: str,
    ):
        self.dataset_name = dataset_name
        pass

    def label_degree(
        self,
        dataset_name: str,
    ):
        pass
