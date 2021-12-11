import torch


def process_mask(mask, num_heads):
    batch_size, N = mask.size()
    mask = mask.view(batch_size, 1, 1, N)
    mask = mask.expand(batch_size, num_heads, N, N)

    return mask
