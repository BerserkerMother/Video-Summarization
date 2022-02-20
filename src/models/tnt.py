import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TNT(nn.Module):
    """
    TNT!!Booom
    Transformer Needs summary Token
    """

    def __init__(self, heads, d_model, num_sumtokens, layers, mask_size, dropout, max_len, device):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.num_sumtokens = num_sumtokens
        self.layers = layers
        self.max_len = max_len
        self.mask_size = mask_size
        self.in_features = 1024
        self.device = device
        self.vid_len = 320

        self.sum_tokens = nn.Parameter(
            torch.zeros(1, self.num_sumtokens, self.d_model))

        self.embedding_layer = Embedding(
            in_features=self.in_features, d_model=self.d_model,
            use_pos=True, sparsity=0.0, use_cls=False)

        self.encoder = Encoder(heads, self.d_model, self.layers, dropout)
        self.decoder = Decoder(self.num_sumtokens, 320, self.d_model)

        self.final_layer = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.Linear(2 * self.d_model, 2)
        )

    def criterian(self, pred, true, representation):
        # mse
        mse = F.mse_loss(pred, true)
        # diversity
        # n = representation.size(1)
        # rep1 = representation.expand(n, n, -1)  # [n, n, dim]
        # rep2 = representation.transpose(0, 1)  # [n, 1, dim]

        # similarity = F.cosine_similarity(rep1, rep2, dim=-1).fill_diagonal_(0)
        # div_loss = similarity.sum() / (n * (n-1))

        # loss = mse + div_loss
        return mse

    def forward(self, x):
        bs, n, _ = x.size()
        x = self.embedding_layer(x)
        x = torch.cat([x, self.sum_tokens], dim=1)

        mem = self.encoder(x, None)
        # sum tokens
        summary_tokens = mem[:, -self.num_sumtokens:]
        summary = self.decoder(summary_tokens)
        # upsample to original
        # nearest neighbor
        token_expand = (n // self.vid_len) + 1
        sum_toks = summary.view(self.vid_len, 1, -1) \
            .expand(self.vid_len, token_expand, self.d_model)
        sum_toks = sum_toks.contiguous().view(1, -1, self.d_model). \
            expand(bs, self.vid_len * token_expand, self.d_model)
        summary = sum_toks[:, :n, :]

        final_out = torch.sigmoid(self.final_layer(summary)).squeeze(-1)

        return final_out, mem[:, 1:]


class Decoder(nn.Module):
    def __init__(self, num_tokens, out_size, d_model):
        super().__init__()
        self.sizes = [num_tokens, 2*num_tokens, 4*num_tokens]
        self.deconv1 = nn.ConvTranspose1d(self.sizes[0], self.sizes[1], 1)
        self.deconv2 = nn.ConvTranspose1d(self.sizes[1], self.sizes[2], 1)
        self.deconv3 = nn.ConvTranspose1d(self.sizes[2], out_size, 1)

        self.sa = MultiHeadAttention(heads=4, d_model=d_model)
        self.out_size = out_size
        self.num_tokens = num_tokens

    def forward(self, x, mem, mode='stepwise'):
        # x = self.sa(x)
        select_idx = []
        for i in self.sizes:
            n = mem.size(1)
            u = (n // i)
            if mode == 'stepwise':
                idx = torch.arange(n, device='cuda')[::u][:i]
            elif mode == 'random':
                idx = np.random.choice(n, size=u)
            select_idx.append(idx)

        x = F.relu(self.deconv1(x+mem[:, select_idx[0]]))
        x = F.relu(self.deconv2(x+mem[:, select_idx[1]]))
        out = self.deconv3(x+mem[:, select_idx[2]])
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
        return x.view(bs, n, -1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
        query = self._split_heads(self.q(q))
        key = self._split_heads(self.k(k))
        value = self._split_heads(self.v(v))

        alignment_score = self._scaled_dot_product(query, key, mask)
        att_weights = self.drop_out(F.softmax(alignment_score, dim=-1))

        out = self._out_proj(att_weights @ value)

        return out, att_weights


class Embedding(nn.Module):
    def __init__(self, in_features, d_model: int = 512,
                 max_len: int = 2000, sparsity: float = 0.5,
                 use_cls: bool = False, use_pos: bool = True):
        super(Embedding, self).__init__()
        # model info
        self.in_features = in_features
        self.d_model = d_model
        self.max_len = max_len
        self.use_cls = use_cls
        self.use_pos = use_pos

        # model layers
        self.feature_transform = nn.Linear(in_features, d_model)
        if use_pos:
            self.positional_encoding = PositionalEncoding(emb_size=d_model,
                                                          maxlen=max_len,
                                                          dropout=sparsity)
        # cls token
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros((1, 1, d_model)))

    def forward(self, x: Tensor):
        batch_size = x.size()[0]

        x = self.feature_transform(x)
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
