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


def load_yaml(path):
    with open(path, 'r') as f:
        try:
            splits = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    return splits


def load_json(path):
    with open(path) as f:
        splits = json.load(f)

    return splits


def mse_with_mask_loss(output, targets, mask, reduction="avg"):
    batch_size, N, _ = output.size()
    output = output.view(batch_size, N)

    scale = torch.ones_like(output)
    scale[mask] = 0.0

    output = output * scale
    targets = targets * scale
    loss = 0.5 * ((output - targets) ** 2)

    if reduction == "avg":
        return loss.mean()
    return loss.sum()
