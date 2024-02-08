from typing import List

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid

from src.iid.models.ML import GraphMLTrain
from src.iid.utils import plot_metrics
from src.iid.utils.constants import MLDefaultSettings
from src.utils.constants import DATA_DIR, REPORT_DIR
from src.utils.logger import get_logger
from src.utils.to_json_file import to_json_file

logger = get_logger()


class AutoencoderDR(nn.Module):
    def __init__(self, input_dim, hidden_dims, encoding_dim):
        super(AutoencoderDR, self).__init__()
        encoder_layers = []
        decoder_layers = []

        # Construct encoder layers
        previous_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(previous_dim, dim))
            encoder_layers.append(nn.ReLU(True))
            previous_dim = dim
        encoder_layers.append(nn.Linear(previous_dim, encoding_dim))
        encoder_layers.append(nn.ReLU(True))

        # Construct decoder layers
        previous_dim = encoding_dim
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(previous_dim, dim))
            decoder_layers.append(nn.ReLU(True))
            previous_dim = dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
        decoder_layers.append(nn.ReLU(True))
        # This can be changed based on the range of your data. For normalized data, you might use nn.Sigmoid()

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TabularEncoder:
    def __init__(
        self, data, hidden_dims: List[int], encoding_dim: int, device: torch.device
    ):
        self.data = data
        self.encoding_dim = encoding_dim
        self.autoencoder = AutoencoderDR(data.shape[1], hidden_dims, encoding_dim)
        self.data.to(device)
        self.autoencoder.to(device)
        self.autoencoder.encoder.to(device)
        self.autoencoder.decoder.to(device)

    def train(self, epochs: int = 200):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)

        batch_size = 16

        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, len(self.data), batch_size):
                batch_x = self.data[i : i + batch_size]

                # Forward pass
                outputs = self.autoencoder(batch_x)
                loss = criterion(outputs, batch_x)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    dataset_name = "PubMed"
    label_dict = {
        0: "Class 1",
        1: "Class 2",
        2: "Class 3",
    }
    data_dir = DATA_DIR / dataset_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_dir = REPORT_DIR / dataset_name / "DR"
    if not report_dir.exists():
        report_dir.mkdir(parents=True)
    dataset = Planetoid(root=data_dir, name=dataset_name)
    dataset.data.to(device)
    logger.info(dataset.data.x.shape[1])
    x = dataset.data.x
    # 900, 300, 100
    output_dim_performances = {}
    for out_dim in range(10, 100, 3):
        dr_auto_encoder = TabularEncoder(
            x,
            hidden_dims=[700, 350, 150],
            encoding_dim=out_dim,
            device=device,
        )
        dr_auto_encoder.train()
        dr_x = dr_auto_encoder.autoencoder.encoder(x).detach().cpu().numpy()
        dataset.data.x = dr_x
        ml_default_settings = MLDefaultSettings()
        feature_1433_df = pd.DataFrame(dr_x)
        feature_1433_df["y"] = dataset.data.y.detach().cpu().numpy()

        feature_1433_model = GraphMLTrain(
            feature_1433_df,
            y_field_name="y",
            svm_best_param=ml_default_settings.pretrain_svm_best_param,
            knn_best_param=ml_default_settings.pretrain_knn_best_param,
            rf_best_param=ml_default_settings.pretrain_rf_best_param,
            risk_labels=label_dict,
        )
        feature_1433_ml_models = feature_1433_model.train(grid_search=False)
        feature_1433_model_metrics = feature_1433_model.plt_3_confusion_matrix(
            feature_1433_ml_models,
            "DR Confusion Matrix",
            report_dir / "DR_confusion_matrix.png",
            plot_it=True,
            balanced_ac=True,
        )
        logger.critical(feature_1433_model_metrics)
        to_json_file(
            feature_1433_model_metrics, report_dir / f"dr_metrics_{out_dim}.json"
        )
        output_dim_performances[out_dim] = feature_1433_model_metrics
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
            metrics=output_dim_performances,
            title="DR Performance vs Dimensionality",
            x_title="Dimensionality",
            y_title=metric_name.upper(),
            metric_name=metric_name,
            filename=report_dir / f"dr-dim-{metric_name}.png",
        )
