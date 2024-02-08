from typing import Optional, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split

from src.traffic.ml_base import MLTrainBase
from src.utils.logger import get_logger

logger = get_logger()


class GraphMLTrain(MLTrainBase):
    RISK_LABELS = [0, 1, 2, 3, 4, 5, 6]

    def __init__(
        self,
        train_data: pd.DataFrame,
        y_field_name: str = "y",
        svm_best_param: Optional[dict] = None,
        knn_best_param: Optional[dict] = None,
        rf_best_param: Optional[dict] = None,
        oversampling: Optional[str] = None,
        test_size: float = 0.3,
        risk_labels: Optional[list] = None,
        *args,  # noqa
        **kwargs,  # noqa
    ):
        super().__init__()
        if risk_labels:
            self.RISK_LABELS = risk_labels
        self.y_field_name = y_field_name
        self.svm_best_param = svm_best_param
        self.knn_best_param = knn_best_param
        self.rf_best_param = rf_best_param

        self.train_data = train_data
        self.y = None
        # init train variable
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # init svc/knn/rf models
        self.svc_model = None
        self.knn_model = None
        self.rf_model = None

        # oversampling
        self.oversampling = oversampling

        # split the train and test dataset
        self.test_size = test_size
        # this will give our whole dataset into
        self.x_train, self.y_train, self.x_test, self.y_test = self.preprocess()

    def preprocess(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        over sample the dataset, and then
        split the train dataset

        Returns
        ------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            which are x_train, y_train, x_test, y_test
        """

        y = self.train_data.pop(self.y_field_name)
        self.y = y
        train_x = self.train_data

        if self.oversampling is None:
            x_train, x_test, y_train, y_test = train_test_split(
                train_x, y, test_size=self.test_size, random_state=42, shuffle=True
            )
            return x_train, y_train, x_test, y_test

        x_train, x_test, y_train, y_test = train_test_split(
            train_x, y, test_size=self.test_size, random_state=42, shuffle=True
        )
        if self.oversampling == "smote":
            random_over_sampler = SMOTE(random_state=42)
        elif self.oversampling == "random":
            random_over_sampler = RandomOverSampler(random_state=42)
        else:
            raise ValueError("oversampling method is not supported")
        x_res, y_res = random_over_sampler.fit_resample(x_train, y_train)
        return x_res, y_res, x_test, y_test
