
from numpy.lib import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, **configs):
        super().__init__()

        if configs['d_model'] is None:
            raise ValueError("model dimension should be specified")

        self.num_heads = configs['num_heads']
        self.d_model = configs['d_model']

        assert self.d_model % self.num_heads == 0
        # according to original paper -> d_key = d_value = d_model // num_heads
        # -> d_model = d_key * num_heads = d_value * num_heads
        self.d_key = self.d_model // self.num_heads
        self.out_head_size = self.d_key * self.num_heads
        self.dropout_rate = configs['dropout_rate']

        self._query = nn.Linear(self.d_model, self.out_head_size)
        self._key = nn.Linear(self.d_model, self.out_head_size)
        self._value = nn.Linear(self.d_model, self.out_head_size)

        self._out = nn.Linear(self.out_head_size, self.d_model)

        self._drop_out = nn.Dropout(self.dropout_rate)

    def _split_heads(self, x):
        # x_shape -> (bs, seq_len, num_heads*d_key)
        # x_new_shape -> (bs, seq_len, num_heads, d_key)
        x_new_shape = x.shape[:-1] + (self.num_heads, self.d_key)
        x = x.view(*x_new_shape)
        return x.permute(0, 2, 1, 3)  # -> (bs, num_heads, seq_len, d_key)

    def _scaled_dot_product(self, q, k, mask):
        scaled_dot_product = torch.matmul(
            q, k.transpose(-1, -2)) / math.sqrt(self.d_key)

        if mask is not None:
            scaled_dot_product += (mask*-1e9)

        return scaled_dot_product

    def _out_projection(self, att):
        bs = att.shape[0]
        seq_len = att.shape[2]
        return att.view(bs, seq_len, -1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None):

        q = self._split_heads(self._query(query))
        k = self._split_heads(self._key(key))
        v = self._split_heads(self._value(value))

        alignment_scores = self._scaled_dot_product(q, k, mask)
        attention_weights = F.softmax(alignment_scores, dim=-1)
        attention = torch.matmul(attention_weights, v)
        # attnetion -> (bs, heads, seq_len, d_value)
        attention = self._drop_out(attention)

        out = self._out(self._out_projection(attention))
        # out -> (bs, seq_len, d_model)

        return out
