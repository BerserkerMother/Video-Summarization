import torch

import yaml
import json
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.num = 0

    def update(self, val, num):
        self.val += val
        self.num += num

    def avg(self):
        return self.val / self.num
