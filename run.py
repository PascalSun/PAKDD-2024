import random

import torch

from src.iid.classification import process
from src.iid.parser import parse_args
from src.utils.logger import get_logger

# set seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

logger = get_logger()

if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    process(args)
