from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

import os
from pathlib import Path

RAW = "raw"
DATA = "data"
REPORTS = "reports"
MODELS = "models"

PROJECT_DIR: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent

TEMP_OUTPUT_DIR = Path("/tmp")

LOG_DIR: Path = PROJECT_DIR / "logs"

DATA_DIR = PROJECT_DIR / DATA

MODEL_DIR = PROJECT_DIR / MODELS

REPORT_DIR = PROJECT_DIR / REPORTS

for folder in [LOG_DIR, DATA_DIR, MODEL_DIR, MODEL_DIR]:
    if not folder.exists():
        folder.mkdir(exist_ok=True, parents=True)


class DataSetEnum(Enum):
    KarateClub = "KarateClub"
    Mitcham = "Mitcham"
    OSM = "OSM"
    PubMed = "PubMed"
    CiteSeer = "CiteSeer"
    Cora = "Cora"
    Travel = "Travel"
    AMAZON_COMPUTERS = "AMAZON_COMPUTERS"
    AMAZON_PHOTO = "AMAZON_PHOTO"
    AMAZON_PRODUCTS = "AMAZON_PRODUCTS"
    NELL = "NELL"
    Reddit = "Reddit"
    CitationsFull_Cora = "CitationsFull_Cora"
    CitationsFull_CiteSeer = "CitationsFull_CiteSeer"
    CitationsFull_PubMed = "CitationsFull_PubMed"
    CitationsFull_Cora_ML = "CitationsFull_Cora_ML"
    CitationsFull_DBLP = "CitationsFull_DBLP"
    Cora_Full = "Cora_Full"
    Coauthor_CS = "Coauther_CS"
    Coauthor_Physics = "Coauther_Physics"
    Flickr = "Flickr"
    Yelp = "Yelp"
    # Entities_AIFB = "Entities_AIFB"
    # Entities_AM = "Entities_AM"
    # Entities_MUTAG = "Entities_MUTAG"
    # Entities_BGS = "Entities_BGS"
    AttributedGraphDataset_Wiki = "AttributedGraphDataset_Wiki"
    AttributedGraphDataset_Cora = "AttributedGraphDataset_Cora"
    AttributedGraphDataset_CiteSeer = "AttributedGraphDataset_CiteSeer"
    AttributedGraphDataset_Pubmed = "AttributedGraphDataset_Pubmed"
    AttributedGraphDataset_BlogCatalog = "AttributedGraphDataset_BlogCatalog"
    AttributedGraphDataset_PPI = "AttributedGraphDataset_PPI"
    AttributedGraphDataset_Flickr = "AttributedGraphDataset_Flickr"
    AttributedGraphDataset_Facebook = "AttributedGraphDataset_Facebook"
    AttributedGraphDataset_TWeibo = "AttributedGraphDataset_TWeibo"
    AttributedGraphDataset_MAG = "AttributedGraphDataset_MAG"

    WEBKB_Cornell = "WEBKB_Cornell"
    WEBKB_Texas = "WEBKB_Texas"
    WEBKB_Wisconsin = "WEBKB_Wisconsin"

    HeterophilousGraphDataset_Roman_empire = "HeterophilousGraphDataset_Roman_empire"
    HeterophilousGraphDataset_Amazon_ratings = (
        "HeterophilousGraphDataset_Amazon_ratings"
    )
    HeterophilousGraphDataset_Minesweeper = "HeterophilousGraphDataset_Minesweeper"
    HeterophilousGraphDataset_Tolokers = "HeterophilousGraphDataset_Tolokers"
    HeterophilousGraphDataset_Questions = "HeterophilousGraphDataset_Questions"

    Actor = "Actor"
    GitHub = "GitHub"

    TWITCH_DE = "TWITCH_DE"
    TWITCH_EN = "TWITCH_EN"
    TWITCH_ES = "TWITCH_ES"
    TWITCH_FR = "TWITCH_FR"
    TWITCH_PT = "TWITCH_PT"
    TWITCH_RU = "TWITCH_RU"

    PolBlogs = "PolBlogs"
    EllipticBitcoinDataset = "EllipticBitcoinDataset"
    DGraphFin = "DGraphFin"

    JODIEDataset_Reddit = "JODIEDataset_Reddit"
    JODIEDataset_Wikipedia = "JODIEDataset_Wikipedia"
    JODIEDataset_MOOC = "JODIEDataset_MOOC"
    JODIEDataset_LastFM = "JODIEDataset_LastFM"


class DataSetModel(BaseModel):
    dataset: DataSetEnum


class ModelsEnum(Enum):
    random_input = "random_input"
    feature_centrality = "feature_centrality"
    feature_1433 = "feature_1433"
    graph_sage = "graph_sage"
    gcn = "gcn"
    node2vec = "node2vec"
    gae = "gae"
    graph_ag = "graph_ag"


class GAEEncoderEnum(Enum):
    gcn = "gcn"
    graph_sage = "graph_sage"


class GAEFeatureEnum(Enum):
    centrality = "centrality"
    feature_1433 = "feature_1433"


class Node2VecParamModeEnum(Enum):
    dim = "dim"
    walk_length = "walk_length"
    walk_per_node = "walk_per_node"
    p = "p"
    q = "q"
    num_negative_samples = "num_negative_samples"


class NNTypeEnum(Enum):
    unsupervised = "unsupervised"
    supervised_centrality = "supervised_centrality"
    supervised_feature = "supervised_feature"


class GraphSAGEAggrEnum(Enum):
    mean = "mean"
    max = "max"


class MLDefaultSettings(BaseModel):
    pretrain_svm_best_param: dict = Field(
        {"C": 10, "gamma": "auto", "kernel": "rbf", "probability": True},
        description="Pretrained SVM best parameters.",
    )
    pretrain_knn_best_param: dict = Field(
        {
            "metric": "euclidean",
            "n_neighbors": 30,
            # "weights": "distance",  # distance or uniform
        },
        description="Pretrained KNN best parameters.",
    )
    pretrain_rf_best_param: dict = Field(
        {
            "bootstrap": True,
            "max_depth": 15,
            "max_features": 0.2,
            "min_samples_leaf": 10,
            "min_samples_split": 10,
            "n_estimators": 200,
            "random_state": 42,
        },
        description="Pretrained Random Forest best parameters.",
    )


class TravelDatasetCityEnum(Enum):
    Miami = "Miami"
    Los_Angeles = "Los Angeles"
    Orlando = "Orlando"
    Dallas = "Dallas"
    Houston = "Houston"
    New_York = "New York"


class TravelDatasetStateEnum(Enum):
    Alabama = "AL"
    Alaska = "AK"
    Arizona = "AZ"
    Arkansas = "AR"
    California = "CA"
    Colorado = "CO"
    Connecticut = "CT"
    Delaware = "DE"
    Florida = "FL"
    Georgia = "GA"
    Hawaii = "HI"
    Idaho = "ID"
    Illinois = "IL"
    Indiana = "IN"
    Iowa = "IA"
    Kansas = "KS"
    Kentucky = "KY"
    Louisiana = "LA"
    Maine = "ME"
    Maryland = "MD"
    Massachusetts = "MA"
    Michigan = "MI"
    Minnesota = "MN"
    Mississippi = "MS"
    Missouri = "MO"
    Montana = "MT"
    Nebraska = "NE"
    Nevada = "NV"
    New_Hampshire = "NH"
    New_Jersey = "NJ"
    New_Mexico = "NM"
    New_York = "NY"
    North_Carolina = "NC"
    North_Dakota = "ND"
    Ohio = "OH"
    Oklahoma = "OK"
    Oregon = "OR"
    Pennsylvania = "PA"
    Rhode_Island = "RI"
    South_Carolina = "SC"
    South_Dakota = "SD"
    Tennessee = "TN"
    Texas = "TX"
    Utah = "UT"
    Vermont = "VT"
    Virginia = "VA"
    Washington = "WA"
    West_Virginia = "WV"
    Wisconsin = "WI"
    Wyoming = "WY"
    District_of_Columbia = "DC"
    American_Samoa = "AS"
    Guam = "GU"
    Northern_Mariana_Islands = "MP"
    Puerto_Rico = "PR"
    United_States_Minor_Outlying_Islands = "UM"
    U_S_Virgin_Islands = "VI"


class TravelDatasetName(BaseModel):
    state: TravelDatasetStateEnum
    city: Optional[TravelDatasetCityEnum]
