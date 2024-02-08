from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger()


def evaluate_model(
    model,
    dataset,
    title: str = None,
    label_dict: Optional[Dict] = None,
    filename: Optional[Path] = None,
    plot_it: bool = False,
) -> dict:
    """
    This is used to evaluate torch NN models.

    :param model:
    :param dataset:
    :param title:
    :param label_dict:
    :param filename:
    :param plot_it:
    :return:
    """
    if label_dict:
        display_labels = [label_dict[key] for key in sorted(label_dict.keys())]
    else:
        display_labels = list(set(dataset.data.y.detach().cpu().numpy()))
    logger.debug(display_labels)
    data = dataset.data
    pred_prob, pred = model(data.x, data.edge_index)
    # confusion matrix all
    cms = [
        {
            "confusion_matrix": confusion_matrix(
                data.y.cpu().detach().numpy(),
                pred.cpu().detach().numpy().argmax(axis=1),
            ),
            "name": "All",
        },
        {
            "confusion_matrix": confusion_matrix(
                data.y[data.train_mask].cpu().detach().numpy(),
                pred[data.train_mask].cpu().detach().numpy().argmax(axis=1),
            ),
            "name": "Train",
        },
        {
            "confusion_matrix": confusion_matrix(
                data.y[data.val_mask].cpu().detach().numpy(),
                pred[data.val_mask].cpu().detach().numpy().argmax(axis=1),
            ),
            "name": "Val",
        },
        {
            "confusion_matrix": confusion_matrix(
                data.y[data.test_mask].cpu().detach().numpy(),
                pred[data.test_mask].cpu().detach().numpy().argmax(axis=1),
            ),
            "name": "Test",
        },
    ]
    plot_confusion_matrix(
        display_labels,
        cms,
        title=(title or "Confusion Matrix"),
        filename=filename,
        plot_it=plot_it,
    )

    metrics = model_metric(
        pred.cpu().detach().numpy().argmax(axis=1),
        dataset,
        pred_prob=F.softmax(pred, dim=1).cpu().detach().numpy(),
    )
    return metrics


def plot_confusion_matrix(
    display_labels: List,
    cms: List,
    title: str,
    filename: str = None,
    plot_it: bool = False,
):
    """
    Plot confusion matrix, this is set to the test/val/train/all split confusion matrix.
    This is same as above for the torch NN models.
    :param display_labels:
    :param cms:
    :param title:
    :param filename:
    :param plot_it:
    :return:
    """
    font = {"size": 26}
    plt.rc("font", **font)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(75, 15),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.1]},
    )

    i = 0
    disp = None
    for cm, ax in zip(cms, axes.flatten()):
        disp = ConfusionMatrixDisplay(
            cm["confusion_matrix"], display_labels=[str(x) for x in display_labels]
        )
        try:
            disp.plot(ax=ax, xticks_rotation=45)
            disp.ax_.set_title(cm["name"].upper())
            disp.im_.colorbar.remove()
            disp.ax_.set_xlabel("Predicted label")
            ax.title.set_text(cm["name"])
            if i != 0:
                disp.ax_.set_ylabel("")
        except Exception as e:
            logger.error(e)
            logger.error(cm)

        i += 1

    fig.colorbar(disp.im_, cax=axes[-1])

    fig.suptitle(title.upper())

    if filename:
        # if filename parent directory does not exist, create it
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)

    if plot_it:
        plt.show()


def model_metric(y_pred, dataset, pred_prob: dict) -> dict:
    """
    This is classic classification metrics, we can use it to evaluate the model performance.

    :param y_pred:
    :param dataset:
    :param pred_prob:

    :return:
    """

    data = dataset.data.to("cpu")

    (
        test_precision,
        test_recall,
        test_f_beta_score,
        test_support,
    ) = precision_recall_fscore_support(
        data.y[data.test_mask],
        y_pred[data.test_mask],
    )

    (
        precision_train,
        recall_train,
        f_beta_score_train,
        support_train,
    ) = precision_recall_fscore_support(
        data.y[data.train_mask], y_pred[data.train_mask]
    )
    test_accuracy = accuracy_score(data.y[data.test_mask], y_pred[data.test_mask])
    train_accuracy = accuracy_score(data.y[data.train_mask], y_pred[data.train_mask])

    # add roc_auc, macro_precision, macro_recall, macro_f_beta_score, micro_precision, micro_recall, micro_f_beta_score
    # if only two classes, use roc_auc_score
    if len(set(data.y[data.test_mask])) == 1:
        roc_auc = -1
        roc_train = -1
    elif len(set(data.y[data.test_mask])) == 2:
        roc_auc = roc_auc_score(data.y[data.test_mask], y_pred[data.test_mask])
        roc_train = roc_auc_score(data.y[data.train_mask], y_pred[data.train_mask])
    else:
        try:
            roc_auc = roc_auc_score(
                data.y[data.test_mask], pred_prob[data.test_mask], multi_class="ovo"
            )

            roc_train = roc_auc_score(
                data.y[data.train_mask], pred_prob[data.train_mask], multi_class="ovo"
            )
        except ValueError:
            roc_auc = None
            roc_train = None

    logger.critical("Test Accuracy: %.2f" % test_accuracy)
    logger.critical("Train Accuracy: %.2f" % train_accuracy)
    logger.critical(
        {
            "precision": test_precision,
            "recall": test_recall,
            "f_beta_score": test_f_beta_score,
            "support": test_support,
            "accuracy_score": test_accuracy,
            "train_accuracy_score": train_accuracy,
            "precision_train": precision_train,
            "recall_train": recall_train,
            "f_beta_score_train": f_beta_score_train,
            "support_train": support_train,
        }
    )
    return {
        "precision_test": test_precision,
        "recall_test": test_recall,
        "f_beta_score_test": test_f_beta_score,
        "support_test": test_support,
        "roc_auc_test": roc_auc,
        "accuracy_score": test_accuracy,
        "accuracy_score_train": train_accuracy,
        "precision_train": precision_train,
        "recall_train": recall_train,
        "f_beta_score_train": f_beta_score_train,
        "support_train": support_train,
        "roc_auc_train": roc_train,
    }
