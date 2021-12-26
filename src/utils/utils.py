import torch

import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class AverageMeter:
    def __init__(self):
        self.sum = 0.
        self.num = 0

    def update(self, value, num):
        self.sum += value
        self.num += num

    def avg(self):
        return self.sum / self.num


def mse_with_mask_loss(output, targets, mask):
    output = output.view(output.size()[0], -1)

    scale = torch.ones_like(output)
    scale[mask] = 0.0

    output = output * scale
    loss = 0.5 * ((output - targets) ** 2)

    return loss.sum()

