import torch
from torch import nn, Tensor
import math


class Embedding(nn.Module):
    def __init__(self, in_features, d_model: int = 512,
                 max_len: int = 2000, sparsity: float = 0.5,
                 use_cls: bool = False, use_pos: bool = True, use_patch: bool = False, patch_size: int = 30):
        super(Embedding, self).__init__()
        # model info
        self.in_features = in_features
        self.d_model = d_model
        self.max_len = max_len
        self.use_cls = use_cls
        self.use_pos = use_pos
        self.use_patch = use_patch
        self.patch_size = patch_size

        # model layers
        self.feature_transform = nn.Linear(in_features, d_model)
        if use_pos:
            self.positional_encoding = PositionalEncoding(emb_size=d_model,
                                                          maxlen=max_len,
                                                          dropout=sparsity)
        # cls token
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros((1, 1, d_model)))

        # patch layer
        self.lin = nn.Linear(self.patch_size*self.d_model, self.d_model)

    def extract_patches(self, x, patch_size):
        bs, len, dim = x.size()
        # split input to specified patches
        x = x.split(patch_size, dim=1)
        zero_pad = len % patch_size
        patch = x[0].view(bs, -1)
        for idx, patch_i in enumerate(x):
            if idx == 0:
                continue
            if patch_i.shape[1] == zero_pad:
                pad_size = patch_size - zero_pad
                patch_i = torch.cat(
                    [patch_i, torch.zeros(bs, pad_size, dim)], dim=1)
            patch = torch.cat([patch, patch_i.view(bs, -1)], dim=1)

        return patch.view(bs, -1, patch_size * dim)

    def forward(self, x: Tensor):
        batch_size = x.size(0)
        x = self.feature_transform(x)
        # patch
        if self.use_patch:
            x = self.extract_patches(x, self.patch_size)
            x = self.lin(x)
        if self.use_pos:
            x = self.positional_encoding(x)
        if self.use_cls:
            cls_token = self.cls_token.expand((batch_size, 1, self.d_model))
            x = torch.cat([cls_token, x], dim=1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 2500):
        super(PositionalEncoding, self).__init__()
        angle = torch.exp(- torch.arange(0, emb_size, 2)
                          * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * angle)
        pos_embedding[:, 1::2] = torch.cos(pos * angle)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding
                            + self.pos_embedding[:, :token_embedding.size()[1]])
