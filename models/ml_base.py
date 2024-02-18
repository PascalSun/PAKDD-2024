import multiprocessing
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models.config import RISK_LABELS
from utils.logger import get_logger
from utils.timer import timer

pd.options.plotting.backend = "plotly"
warnings.filterwarnings("ignore")

logger = get_logger()


class MLTrainBase:
    """
    Base class for inherit
    This will implement the base functions for train svm/knn/rf
    """

    # classification labels
    RISK_LABELS = RISK_LABELS

    def __init__(self, *arg, **kwargs):
        """
        Load the embeddings.tsv data, and then match them to the proper asset
        """
        self.svm_best_param = None
        self.knn_best_param = None
        self.rf_best_param = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # init models
        self.svc_model = None
        self.knn_model = None
        self.rf_model = None
        self.n_jobs = multiprocessing.cpu_count()

        """
        Example

        ```
        self.raw_train_data = self.iid_df_without_null.copy()
        self.train_data = self.raw_train_data.copy()

        self.x_train, self.y_train, self.x_test, self.y_test = self.preprocess()
        ```
        """

    def train_svc(self, best_params: dict) -> sklearn.pipeline.Pipeline:
        """
        SVC model to train

        Parameters
        ----------
        best_params: dict
            best params already searched, will be feed to here

        Raises
        ------
        RuntimeError
            something wrong

        Returns:
        --------
        sklearn.pipeline.Pipeline
            model pipeline
        """
        svc_clf = make_pipeline(StandardScaler(), SVC(**best_params))
        svc_clf.fit(self.x_train, self.y_train)
        return svc_clf

    def train_knn(self, best_params) -> KNeighborsClassifier:
        """
        knn model to be trained with provided params

        Parameters
        ----------
        best_params: dict
            best params already searched, will be feed to here

        Raises
        ------
        RuntimeError
            something wrong

        Returns:
        --------
        KNeighborsClassifier
            model pipeline
        """
        neigh = KNeighborsClassifier(**best_params)
        neigh.fit(self.x_train, self.y_train)
        return neigh

    def train_random_forest(self, best_params) -> RandomForestClassifier:
        """
        random forest model to be trained with provided params

        Parameters
        ----------
        best_params: dict
            best params already searched, will be feed to here

        Raises
        ------
        RuntimeError
            something wrong

        Returns:
        --------
        RandomForestClassifier
            model pipeline
        """
        dt_clf = RandomForestClassifier(**best_params)
        dt_clf.fit(self.x_train, self.y_train)
        return dt_clf

    def tune_svc(self, tuned_parameters: Optional[list] = None) -> dict:
        """
        This is the function used to tune the svc model, search for best params

        Parameters
        ----------
        tuned_parameters:  Optional[list], default=None
            a list of params will be used in the search

        Raises
        ------
        RuntimeError
            something wrong

        Returns:
        ---------
        dict
            best params we found

        Search result of this is
        {
            "C": 1000,
            "gamma": "auto",
            "kernel": "rbf"
        }
        """
        # Set the parameters by cross-validation
        # TODO: set multi process
        if not tuned_parameters:
            tuned_parameters = [
                {
                    "kernel": ["rbf"],
                    "gamma": [1e-3, 1e-4, "auto"],
                    "C": [1, 10, 100, 1000],
                },
                # {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
            ]

        # think about get it a better way to evaluate the best params
        scores = ["recall"]
        best_params = None
        for score in scores:
            logger.info("# Tuning hyper-parameters for %s" % score)
            clf = make_pipeline(
                StandardScaler(),
                GridSearchCV(
                    SVC(),
                    param_grid=tuned_parameters,
                    scoring="%s_macro" % score,
                    n_jobs=self.n_jobs,
                ),
            )
            clf.fit(self.x_train, self.y_train)

            logger.info("Best parameters set found on development set:")

            logger.info(clf["gridsearchcv"].best_params_)
            best_params = clf["gridsearchcv"].best_params_

            logger.info("Grid scores on development set:")

            means = clf["gridsearchcv"].cv_results_["mean_test_score"]
            stds = clf["gridsearchcv"].cv_results_["std_test_score"]

            for mean, std, params in zip(
                    means, stds, clf["gridsearchcv"].cv_results_["params"]
            ):
                logger.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            logger.info("Detailed classification report:")
            logger.info("The model is trained on the full development set.")
            logger.info("The scores are computed on the full evaluation set.")
            y_true, y_pred = self.y_test, clf["gridsearchcv"].predict(self.x_test)
            logger.info(classification_report(y_true, y_pred))
        return best_params

    def tune_knn(self, tuned_parameters: Optional[list] = None) -> dict:
        """
        Tune KNN to find out the best params

        Parameters
        ----------
        tuned_parameters:  Optional[list], default=None
            a list of params will be used in the search

        Raises
        ------
        RuntimeError
            something wrong

        Returns:
        ---------
        dict
            best params we found
        """
        if not tuned_parameters:
            tuned_parameters = {
                "n_neighbors": [11, 15, 19, 20],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            }

        clf = make_pipeline(
            StandardScaler(),
            GridSearchCV(
                KNeighborsClassifier(),
                param_grid=tuned_parameters,
                n_jobs=self.n_jobs,
                scoring="recall_macro",
            ),
        )

        clf.fit(self.x_train, self.y_train)
        return clf["gridsearchcv"].best_params_

    def tune_random_forest(self, tuned_parameters: Optional[list] = None) -> dict:
        """
        Tune random forest

        Parameters
        ----------
        tuned_parameters:  Optional[list], default=None
            a list of params will be used in the search

        Raises
        ------
        RuntimeError
            something wrong

        Returns:
        ---------
        dict
            best params we found
        """
        if not tuned_parameters:
            tuned_parameters = {
                "bootstrap": [True],
                "max_depth": [5, 10, 20],  # focus on others first
                "max_features": [
                    0.2,
                    0.5,
                    0.8,
                ],  # can be percentage for the features, this one is more important
                "min_samples_leaf": [
                    10,
                ],  # FIXME: should not be less than the sample split, maybe do not need this
                "min_samples_split": [10],
                "n_estimators": [200],
                "random_state": [42],
            }
        rf = RandomForestClassifier()

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=tuned_parameters,
            cv=3,
            n_jobs=self.n_jobs,
            verbose=2,
        )
        grid_search.fit(self.x_train, self.y_train)
        return grid_search.best_params_

    def plt_3_confusion_matrix(
            self,
            models: list,
            title: str,
            filename: Optional[Path] = None,
            plot_it: bool = True,
            balanced_ac: bool = False,
    ) -> list:
        """
        plot three confusion matrix in the same image for SVM/KNN/RF

        Parameters
        ----------
        models:  list
            a list of machine learning model
        title: str
            figure title

        filename: Path, default=None
            file path to save the figure.
        plot_it: bool, default=True
        balanced_ac: bool, default=False
        Raises
        ------
        RuntimeError

            something wrong
        Returns:
        ---------
        list
            model metrics to be returned as a list
            ```
            [{
                "precision": precision,
                "recall": recall,
                "f_beta_score": f_beta_score,
                "support": support,
                "accuracy_score": accuracy,
                "train_accuracy_score": train_accuracy,
                "precision_train": precision_train,
                "recall_train": recall_train,
                "f_beta_score_train": f_beta_score_train,
                "support_train": support_train,
            }, xxx]
            ```
        """
        font = {"size": 26}
        plt.rc("font", **font)
        fig, axes = plt.subplots(
            nrows=1,
            ncols=4,
            figsize=(55, 15),
            gridspec_kw={"width_ratios": [1.3, 1.3, 1.3, 0.1]},
        )
        display_labels = self.RISK_LABELS
        logger.info(display_labels)
        i = 0
        disp = None
        for clf, ax in zip(models, axes.flatten()):
            y_pred = clf["model"].predict(self.x_test)
            cf_matrix = confusion_matrix(
                self.y_test, y_pred, labels=list(range(len(display_labels)))
            )
            disp = ConfusionMatrixDisplay(cf_matrix, display_labels=display_labels)
            disp.plot(ax=ax, xticks_rotation=45)
            disp.ax_.set_title(clf["name"].upper())
            disp.im_.colorbar.remove()
            disp.ax_.set_xlabel("Predicted label")
            ax.title.set_text(clf["name"])
            if i != 0:
                disp.ax_.set_ylabel("")
            i += 1

        fig.colorbar(disp.im_, cax=axes[-1])

        fig.suptitle(title.upper())

        if filename:
            plt.savefig(filename)

        if plot_it:
            plt.show()

        models_metric = []
        for clf, ax in zip(models, axes.flatten()):
            logger.info(f"{title}: {clf['name']}")
            model_metric = {
                "name": clf["name"],
                **self.model_metric(clf["model"], balanced_ac=balanced_ac),
                "best_params": clf["best_params"],
            }
            models_metric.append(model_metric)
        return models_metric

    def model_metric(self, clf, balanced_ac: bool = False) -> dict:
        """
        calculate the model metrics, return as a dict

        Parameters
        ----------
        clf:  model
            the model to be evaluated
        balanced_ac: bool, default=False

        Raises
        ------
        RuntimeError

            something wrong
        Returns:
        ---------
        dict
            model metrics to be returned as a list
            ```
            {
                "precision": precision,
                "recall": recall,
                "f_beta_score": f_beta_score,
                "support": support,
                "accuracy_score": accuracy,
                "train_accuracy_score": train_accuracy,
                "precision_train": precision_train,
                "recall_train": recall_train,
                "f_beta_score_train": f_beta_score_train,
                "support_train": support_train,
            }
            ```

        """
        y_pred = clf.predict(self.x_test)
        y_pred_train = clf.predict(self.x_train)

        precision, recall, f_beta_score, support = precision_recall_fscore_support(
            self.y_test, y_pred
        )
        (
            macro_precision,
            macro_recall,
            macro_f_beta_score,
            _,
        ) = precision_recall_fscore_support(self.y_test, y_pred, average="macro")
        (
            micro_precision,
            micro_recall,
            micro_f_beta_score,
            _,
        ) = precision_recall_fscore_support(self.y_test, y_pred, average="micro")

        (
            precision_train,
            recall_train,
            f_beta_score_train,
            support_train,
        ) = precision_recall_fscore_support(self.y_train, y_pred_train)
        (
            macro_precision_train,
            macro_recall_train,
            macro_f_beta_score_train,
            _,
        ) = precision_recall_fscore_support(self.y_train, y_pred_train, average="macro")
        (
            micro_precision_train,
            micro_recall_train,
            micro_f_beta_score_train,
            _,
        ) = precision_recall_fscore_support(self.y_train, y_pred_train, average="micro")

        # calculate accuracy
        if balanced_ac:
            accuracy = balanced_accuracy_score(self.y_test, y_pred)
            train_accuracy = balanced_accuracy_score(self.y_train, y_pred_train)
        else:
            accuracy = accuracy_score(self.y_test, y_pred)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
        logger.critical("Test Accuracy: %.2f" % accuracy)
        logger.critical("Train Accuracy: %.2f" % train_accuracy)
        logger.info(self.y_test.shape)
        logger.info(clf.predict_proba(self.x_test).shape)

        if len(list(set(self.y_test))) == 2:
            roc_auc = roc_auc_score(self.y_test, clf.predict_proba(self.x_test)[:, 1])
            roc_auc_train = roc_auc_score(
                self.y_train, clf.predict_proba(self.x_train)[:, 1]
            )
        else:
            import numpy as np

            logger.info(np.unique(self.y_test))
            logger.info(list(range(len(self.RISK_LABELS))))
            if len(np.unique(self.y_test)) == 1:
                logger.warning(
                    "Warning: Only one unique class in y_test. Can't compute ROC AUC for test set."
                )
                roc_auc = None
            else:
                try:
                    roc_auc = roc_auc_score(
                        self.y_test,
                        clf.predict_proba(self.x_test),
                        multi_class="ovr",
                        labels=list(range(len(self.RISK_LABELS))),
                    )
                except Exception as e:
                    logger.error(e)
                    roc_auc = None

            if len(np.unique(self.y_train)) == 1:
                logger.warning(
                    "Warning: Only one unique class in y_train. Can't compute ROC AUC for training set."
                )
                roc_auc_train = None
            else:
                roc_auc_train = roc_auc_score(
                    self.y_train,
                    clf.predict_proba(self.x_train),
                    multi_class="ovr",
                    labels=list(range(len(self.RISK_LABELS))),
                )

        metrics = {
            "precision": precision,
            "recall": recall,
            "f_beta_score": f_beta_score,
            "support": support,
            "accuracy_score": accuracy,
            "roc_auc": roc_auc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f_beta_score": macro_f_beta_score,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f_beta_score": micro_f_beta_score,
            "precision_train": precision_train,
            "recall_train": recall_train,
            "f_beta_score_train": f_beta_score_train,
            "support_train": support_train,
            "macro_precision_train": macro_precision_train,
            "macro_recall_train": macro_recall_train,
            "macro_f_beta_score_train": macro_f_beta_score_train,
            "micro_precision_train": micro_precision_train,
            "micro_recall_train": micro_recall_train,
            "micro_f_beta_score_train": micro_f_beta_score_train,
            "accuracy_score_train": train_accuracy,
            "roc_auc_train": roc_auc_train,
        }
        logger.info(metrics)
        return metrics

    def train(
            self,
            grid_search: bool = False,
            svm_param: Optional[dict] = None,
            knn_param: Optional[dict] = None,
            rf_param: Optional[dict] = None,
            *args,
            **kwargs,
    ) -> list:
        """
        main function will be called when we call train in command line

        Parameters
        ----------
        grid_search:  bool
            doing grid search or not, if False, then will train model directly from tuned params
        svm_param: Optional[dict]
            svm param used to try
        knn_param: Optional[dict]
            knn param used to try
        rf_param: Optional[dict]
            rf param to try

        """
        # SVM
        if grid_search:
            with timer(logger, "tune svc"):
                svc_best_param = self.tune_svc()
            logger.critical(f"The best params SVM {svc_best_param}")
            self.svc_model = self.train_svc(svc_best_param)

            with timer(logger, "tune knn"):
                knn_best_param = self.tune_knn()
            logger.critical(f"The best params for KNN {knn_best_param}")
            self.knn_model = self.train_knn(knn_best_param)

            with timer(logger, "tune random forest"):
                rf_best_param = self.tune_random_forest()
            logger.critical(f"The best params for Random Forest {rf_best_param}")
            self.rf_model = self.train_random_forest(rf_best_param)
        else:
            # SVC
            with timer(logger, "train svc"):
                svc_best_param = svm_param or self.svm_best_param
                if svc_best_param:
                    self.svc_model = self.train_svc(svc_best_param)
                else:
                    logger.warning("have not set the svm best param")

            with timer(logger, "train knn"):
                # KNN model
                knn_best_param = knn_param or self.knn_best_param
                if knn_best_param:
                    self.knn_model = self.train_knn(knn_best_param)
                else:
                    logger.warning("have not set knn best param")

            with timer(logger, "train random forest"):
                # RF model
                rf_best_param = rf_param or self.rf_best_param
                if rf_best_param:
                    self.rf_model = self.train_random_forest(rf_best_param)
                else:
                    logger.warning("have not set rf best param")

        models = [
            {"name": "SVM", "model": self.svc_model, "best_params": svc_best_param},
            {"name": "KNN", "model": self.knn_model, "best_params": knn_best_param},
            {"name": "RF", "model": self.rf_model, "best_params": rf_best_param},
        ]
        return models

    def preprocess(self):
        """
        Not yet implemented
        """
        logger.warning("not yet implemented preprocess")
        return None, None, None, None
