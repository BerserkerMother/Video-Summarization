import torch
import numpy as np
from torch import nn, Tensor
import math


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        angle = torch.exp(- torch.arange(0, emb_size, 2)
                          * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * angle)
        pos_embedding[:, 1::2] = torch.cos(pos * angle)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class MSELossWithMask():
    def __call__(self, pred, true):
        output = pred.view(pred.size()[0], -1)

        scale = torch.ones_like(output)
        scale[true == -1] = 0.0

        pred = pred * scale
        loss = (pred - true) ** 2

        return 0.5 * loss.sum()
