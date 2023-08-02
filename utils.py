import os
import random
from typing import NoReturn

import numpy as np
import torch


def setup_seed(seed: int) -> NoReturn:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
