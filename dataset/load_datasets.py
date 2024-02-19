from torch_geometric.data import Dataset
from torch_geometric.datasets import (  # Entities
    NELL,
    Actor,
    Amazon,
    AmazonProducts,
    AttributedGraphDataset,
    CitationFull,
    Coauthor,
    CoraFull,
    DGraphFin,
    EllipticBitcoinDataset,
    Flickr,
    GitHub,
    HeterophilousGraphDataset,
    JODIEDataset,
    KarateClub,
    Planetoid,
    PolBlogs,
    Reddit,
    Twitch,
    WebKB,
    Yelp,
)

from graph_metrics.constants import DATASET_LABELS_MAPPING
from utils.constants import DataSetEnum
from utils.constants import DATA_DIR
from utils.logger import get_logger

IGNORED_DATASET = [
    DataSetEnum.AttributedGraphDataset_TWeibo.value,
    DataSetEnum.AttributedGraphDataset_MAG.value,
]

logger = get_logger()


def load_dataset(dataset_name: str) -> Dataset:  # noqa
    data_dir = DATA_DIR / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset_name in IGNORED_DATASET:
        raise ValueError(f"Dataset {dataset_name} is not included")
    if dataset_name in [
        DataSetEnum.Cora.value,
        DataSetEnum.PubMed.value,
        DataSetEnum.CiteSeer.value,
    ]:
        dataset = Planetoid(root=data_dir, name=dataset_name)
    elif dataset_name == DataSetEnum.KarateClub.value:
        dataset = KarateClub()
        dataset.data.name = DataSetEnum.KarateClub.value
    elif dataset_name == DataSetEnum.AMAZON_COMPUTERS.value:
        dataset = Amazon(root=data_dir, name="Computers")
    elif dataset_name == DataSetEnum.AMAZON_PHOTO.value:
        dataset = Amazon(root=data_dir, name="Photo")
    elif dataset_name == DataSetEnum.NELL.value:
        dataset = NELL(root=data_dir)
    elif dataset_name == DataSetEnum.Reddit.value:
        dataset = Reddit(root=data_dir)
        dataset.name = DataSetEnum.Reddit.value
    elif dataset_name == DataSetEnum.AMAZON_PRODUCTS.value:
        dataset = AmazonProducts(root=data_dir)
    elif dataset_name in [
        DataSetEnum.CitationsFull_Cora.value,
        DataSetEnum.CitationsFull_CiteSeer.value,
        DataSetEnum.CitationsFull_PubMed.value,
        DataSetEnum.CitationsFull_Cora_ML.value,
        DataSetEnum.CitationsFull_DBLP.value,
    ]:
        dataset = CitationFull(root=data_dir, name=dataset_name.split("_")[1].lower())
    elif dataset_name == DataSetEnum.Cora_Full.value:
        dataset = CoraFull(root=data_dir)
    elif dataset_name == DataSetEnum.Coauthor_CS.value:
        dataset = Coauthor(root=data_dir, name="CS")
    elif dataset_name == DataSetEnum.Coauthor_Physics.value:
        dataset = Coauthor(root=data_dir, name="Physics")
    elif dataset_name == DataSetEnum.Flickr.value:
        dataset = Flickr(root=data_dir)
    elif dataset_name == DataSetEnum.Yelp.value:
        dataset = Yelp(root=data_dir)
    elif dataset_name in [
        DataSetEnum.AttributedGraphDataset_Wiki.value,
        DataSetEnum.AttributedGraphDataset_Cora.value,
        DataSetEnum.AttributedGraphDataset_CiteSeer.value,
        DataSetEnum.AttributedGraphDataset_Pubmed.value,
        DataSetEnum.AttributedGraphDataset_BlogCatalog.value,
        DataSetEnum.AttributedGraphDataset_PPI.value,
        DataSetEnum.AttributedGraphDataset_Flickr.value,
        DataSetEnum.AttributedGraphDataset_Facebook.value,
        DataSetEnum.AttributedGraphDataset_TWeibo.value,
        DataSetEnum.AttributedGraphDataset_MAG.value,
    ]:
        dataset = AttributedGraphDataset(
            root=data_dir, name=dataset_name.split("_")[1].lower()
        )

    elif dataset_name in [
        DataSetEnum.WEBKB_Cornell.value,
        DataSetEnum.WEBKB_Texas.value,
        DataSetEnum.WEBKB_Wisconsin.value,
    ]:
        dataset = WebKB(root=data_dir, name=dataset_name.split("_")[1].lower())
    elif dataset_name in [
        DataSetEnum.HeterophilousGraphDataset_Roman_empire.value,
        DataSetEnum.HeterophilousGraphDataset_Amazon_ratings.value,
        DataSetEnum.HeterophilousGraphDataset_Minesweeper.value,
        DataSetEnum.HeterophilousGraphDataset_Tolokers.value,
        DataSetEnum.HeterophilousGraphDataset_Questions.value,
    ]:
        dataset = HeterophilousGraphDataset(
            root=data_dir, name=dataset_name.split("_", 1)[1].replace("_", "-").lower()
        )
    elif dataset_name == DataSetEnum.Actor.value:
        dataset = Actor(root=data_dir)
    elif dataset_name == DataSetEnum.GitHub.value:
        dataset = GitHub(root=data_dir)
    elif dataset_name in [
        DataSetEnum.TWITCH_DE.value,
        DataSetEnum.TWITCH_EN.value,
        DataSetEnum.TWITCH_ES.value,
        DataSetEnum.TWITCH_FR.value,
        DataSetEnum.TWITCH_PT.value,
        DataSetEnum.TWITCH_RU.value,
    ]:
        dataset = Twitch(root=data_dir, name=dataset_name.split("_")[1])
    elif dataset_name == DataSetEnum.PolBlogs.value:
        dataset = PolBlogs(root=data_dir)
    elif dataset_name == DataSetEnum.EllipticBitcoinDataset.value:
        dataset = EllipticBitcoinDataset(root=data_dir)
    elif dataset_name == DataSetEnum.DGraphFin.value:
        dataset = DGraphFin(root=data_dir)
    elif dataset_name in [
        DataSetEnum.JODIEDataset_Reddit.value,
        DataSetEnum.JODIEDataset_Wikipedia.value,
        DataSetEnum.JODIEDataset_MOOC.value,
        DataSetEnum.JODIEDataset_LastFM.value,
    ]:
        dataset = JODIEDataset(root=data_dir, name=dataset_name.split("_")[1].lower())

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")
    dataset.data.labels = DATASET_LABELS_MAPPING.get(dataset_name, None)
    # if dataset.data do not have labels as attribute, then add it
    if not hasattr(dataset.data, "labels"):
        try:
            dataset.data.labels = {i: i for i in range(dataset.data.y.max().item() + 1)}
        except Exception as e:
            logger.debug(f"Error while setting labels for dataset {dataset_name}")
            logger.debug(e)
    return dataset
