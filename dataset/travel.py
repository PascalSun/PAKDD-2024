import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url

from utils.logger import get_logger

logger = get_logger(logger_name="root")


class TRAVELDataset(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = "https://github.com/baixianghuang/travel/raw/main/TAP-city/{}.npz"

    # url = 'https://github.com/baixianghuang/travel/raw/main/TAP-state/{}.npz'

    def __init__(
        self,
        root: str,
        name: str,
        task: str = "occur",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        """

        :param root: where to store the dataset
        :param name: the name of the specific dataset, like "miami_fl", "oriando_fl"
        :param task: doing the prediction of "occur" or "severity"
        :param transform:
        :param pre_transform:
        """
        self.name = name.lower()
        self.task = task.lower()

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.task == "occur":
            self.data.y = self.data.occur_labels
        elif self.task == "severity":
            self.data.y = self.data.severity_labels
        else:
            raise NotImplementedError

    @property
    def raw_dir(self) -> str:
        if len(self.name.split("_")) == 1:
            return osp.join(self.root, "state")
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return f"{self.name}.npz"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        # if len(self.name.split("_")) == 1:
        #     return
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        data = self.read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        # add train_mask, val_mask, test_mask
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[: int(data.num_nodes * 0.8)] = 1
        data.val_mask[int(data.num_nodes * 0.8) : int(data.num_nodes * 0.9)] = 1
        data.test_mask[int(data.num_nodes * 0.9) :] = 1

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}Full()"

    def read_npz(self, path):
        with np.load(path, allow_pickle=True) as f:
            return self.parse_npz(f, self.task)

    @staticmethod
    def parse_npz(f, task):
        crash_time = f["crash_time"]
        x = torch.from_numpy(f["x"]).to(torch.float)
        coords = torch.from_numpy(f["coordinates"]).to(torch.float)
        edge_attr = torch.from_numpy(f["edge_attr"]).to(torch.float)
        cnt_labels = torch.from_numpy(f["cnt_labels"]).to(torch.long)
        occur_labels = torch.from_numpy(f["occur_labels"]).to(torch.long)
        edge_attr_dir = torch.from_numpy(f["edge_attr_dir"]).to(torch.float)
        edge_attr_ang = torch.from_numpy(f["edge_attr_ang"]).to(torch.float)
        severity_labels = torch.from_numpy(f["severity_8labels"]).to(torch.long)
        edge_index = torch.from_numpy(f["edge_index"]).to(torch.long).t().contiguous()
        if task == "occur":
            y = occur_labels
        elif task == "severity":
            y = severity_labels
        else:
            raise NotImplementedError
        logger.info(task)
        logger.info(f"y: {list(set(y.detach().cpu().numpy()))}")
        return Data(
            x=x,
            y=y,
            severity_labels=severity_labels,
            occur_labels=occur_labels,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_attr_dir=edge_attr_dir,
            edge_attr_ang=edge_attr_ang,
            coords=coords,
            cnt_labels=cnt_labels,
            crash_time=crash_time,
        )
