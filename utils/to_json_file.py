import json
from pathlib import Path
from typing import Union

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def to_json_file(data: Union[list, dict], file_path: Path):
    """
    Write data to json file

    :param data:
    :param file_path:
    :return:
    """

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
