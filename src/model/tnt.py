import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model import Embedding


class TNT(nn.Module):
    """
    TNT!!Booom
    Transformer Needs summary Token
    """

    def __init__(self, heads, d_model, num_sumtokens, layers, dropout, max_len, device):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.num_sumtokens = num_sumtokens
        self.layers = layers
        self.max_len = max_len
        self.in_features = 1024
        self.device = device
        self.vid_len = 320
        self.patch_size = 30

        self.sum_tokens = nn.Parameter(
            torch.zeros(1, self.num_sumtokens, self.d_model))

        self.embedding_layer = Embedding(
            in_features=self.in_features, d_model=self.d_model,
            use_pos=True, sparsity=0.0, use_cls=False, use_patch=True, patch_size=self.patch_size)

        self.encoder = Encoder(heads, self.d_model, self.layers, dropout)
        self.deconv = Decov(self.num_sumtokens, self.vid_len)

        self.final_layer = nn.Linear(self.d_model, 1)

    def criterian(self, pred, true):
        # mse
        mse = F.mse_loss(pred, true)
        return mse

    def upsample(self, tokens, num_tokens, n, bs):
        token_expand = (n // num_tokens) + 1
        sum_toks = tokens.view(num_tokens, 1, -1) \
            .expand(num_tokens, token_expand, self.d_model)
        sum_toks = sum_toks.contiguous().view(1, -1, self.d_model). \
            expand(bs, num_tokens * token_expand, self.d_model)
        sum_toks = sum_toks[:, :n, :]

        return sum_toks

    def forward(self, x):
        bs, n, _ = x.size()
        x = self.embedding_layer(x)
        x = torch.cat([x, self.sum_tokens], dim=1)

        mem = self.encoder(x, None)
        # sum tokens
        summary_tokens = mem[:, -self.num_sumtokens:]
        summary = self.deconv(summary_tokens)
        # upsample to original
        # nearest neighbor
        summary = self.upsample(summary, self.vid_len, n, bs)

        final_out = torch.sigmoid(self.final_layer(summary)).squeeze(-1)

        return final_out


class Decov(nn.Module):
    def __init__(self, num_tokens, out_size):
        super().__init__()
        self.sizes = [num_tokens, 2*num_tokens]
        self.deconv1 = nn.ConvTranspose1d(self.sizes[0], self.sizes[1], 1)
        self.deconv2 = nn.ConvTranspose1d(self.sizes[1], out_size, 1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        out = F.relu(self.deconv2(x))
        return out


class Encoder(nn.Module):
    def __init__(self, heads, d_model, enc_layers, dropout):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.enc_layers = enc_layers

        modules = []
        for _ in range(self.enc_layers):
            modules.append(
                EncoderBlock(heads=self.heads, d_model=self.d_model, drop_rate=dropout))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x: Tensor, local_mask):
        for block in self.module_list:
            x = block(x, local_mask)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, heads, d_model, drop_rate=0.3):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.drop_rate = drop_rate

        self.sa = MultiHeadAttention(heads, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.d_model * 2, self.d_model)
        )

    def forward(self, x: Tensor, local_mask):
        attented_x, w = self.sa(q=x, k=x, v=x, mask=local_mask)
        z = self.norm1(attented_x + x)  # residual

        mlp_out = self.mlp(z)
        z2 = self.norm2(mlp_out + z)  # residual

        return z2


class MultiHeadAttention(nn.Module):
    def __init__(self, heads=4, d_model=128):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        assert self.d_model % self.heads == 0
        self.d_k = self.d_model // self.heads

        self.q = nn.Linear(self.d_model, self.d_k * self.heads)
        self.k = nn.Linear(self.d_model, self.d_k * self.heads)
        self.v = nn.Linear(self.d_model, self.d_k * self.heads)

        self.drop_out = nn.Dropout(0.2)

        self.out_proj = nn.Linear(self.d_k * self.heads, self.d_model)

    def _scaled_dot_product(self, q, k, mask=None):
        scaled_dp = (q @ k.transpose(-1, -2)) / self.d_k ** 0.5
        if mask is not None:
            scaled_dp.masked_fill_(mask, float("-inf"))

        return scaled_dp

    def _split_heads(self, x):
        bs, n, h_dk = x.shape
        new_size = (bs, self.heads, n, self.d_k)
        return x.view(*new_size)

    def _out_proj(self, x):
        bs, split, n, dim = x.shape
        return self.out_proj(x.view(bs, n, -1))

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
        query = self._split_heads(self.q(q))
        key = self._split_heads(self.k(k))
        value = self._split_heads(self.v(v))

        alignment_score = self._scaled_dot_product(query, key, mask)
        att_weights = self.drop_out(F.softmax(alignment_score, dim=-1))

        out = self._out_proj(att_weights @ value)

        return out, att_weights
