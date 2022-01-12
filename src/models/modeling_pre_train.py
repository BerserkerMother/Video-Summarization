from typing import Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MAE(nn.Module):
    def __init__(self, heads, d_enc, d_dec, enc_layers, dec_layers, max_len, device):
        super().__init__()
        self.heads = heads
        self.d_enc = d_enc
        self.d_dec = d_dec
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.max_len = max_len
        self.in_features = 1024
        self.device = device

        self.enc_pos_encoding = PositionalEncoding(
            self.max_len, self.d_enc, device)
        self.dec_pos_encoding = PositionalEncoding(
            self.max_len, self.d_dec, device)

        self.mask_token = nn.Parameter(torch.zeros(1, self.d_dec))

        self.first_layer = nn.Linear(self.in_features, self.d_enc)

        self.encoder = Encoder(heads, d_enc, enc_layers)
        self.enc_to_dec = nn.Linear(self.d_enc, self.d_dec)
        self.decoder = Decoder(heads, d_dec, dec_layers)

        self.final_layer = nn.Linear(self.d_dec, self.in_features)

    def pre_train_criterian(self, pred, target, mode):
        if mode == 'Sim':
            pass
        else:
            x = F.normalize(pred, dim=-1, p=2)
            y = F.normalize(target, dim=-1, p=2)
            return (2 - 2 * (x * y).sum(dim=-1)).mean()

    def forward(self, x, mask_idx, vis_idx):
        x = self.first_layer(x)
        x = x[:, vis_idx, :] + \
            torch.index_select(self.enc_pos_encoding(), 0, vis_idx)
        enc_out = self.encoder(x)

        enc_to_dec = self.enc_to_dec(enc_out)

        bs, n, dim = enc_to_dec.shape

        # select the corresponding mask indces from pos_enc and add them to mask token
        x_mask = self.mask_token + \
            torch.index_select(self.dec_pos_encoding(), 0, mask_idx)
        x_mask = x_mask.expand(bs, -1, -1)
        # select the corresponding visible indeces from pos_enc and add them to encoder out
        x_vis = enc_to_dec + \
            torch.index_select(self.dec_pos_encoding(), 0, vis_idx)

        full_x = torch.cat([x_vis, x_mask], dim=1)

        reconst = self.final_layer(self.decoder(full_x))

        return reconst[:, -mask_idx.size(0):]


class Encoder(nn.Module):
    def __init__(self, heads, d_enc, enc_layers):
        super().__init__()
        self.heads = heads
        self.d_dec = d_enc
        self.enc_layers = enc_layers

        modules = []
        for _ in range(self.enc_layers):
            modules.append(
                Block(heads=self.heads, d_model=self.d_dec, drop_rate=0.3))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x: Tensor):
        for block in self.module_list:
            x = block(x)

        return x


class Decoder(nn.Module):
    def __init__(self, heads, d_dec, dec_layers):
        super().__init__()
        self.heads = heads
        self.d_dec = d_dec
        self.dec_layers = dec_layers

        modules = []
        for _ in range(self.dec_layers):
            modules.append(
                Block(heads=self.heads, d_model=self.d_dec, drop_rate=0.3))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x: Tensor):
        for block in self.module_list:
            x = block(x)

        return x


class Block(nn.Module):
    def __init__(self, heads, d_model, drop_rate=0.3):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.drop_rate = drop_rate

        self.sa = MultiHeadAttention(heads, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.d_model*2, self.d_model)
        )

    def forward(self, x: Tensor):
        attented_x, w = self.sa(q=x, k=x, v=x, mask=None)
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

        self.out_proj = nn.Linear(self.d_k*self.heads, self.d_model)

    def _scaled_dot_product(self, q, k, mask=None):
        scaled_dp = (q @ k.transpose(-1, -2)) / self.d_k ** 0.5
        if mask is not None:
            scaled_dp.mask_fill_(mask, -1e9)

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


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, device):
        super().__init__()
        # pos is the position
        pos = torch.arange(max_len, device=device).unsqueeze(dim=1)
        # i is the dimension
        i = torch.arange(d_model, device=device)
        self.d_model = d_model
        # for each dimension of d_model compute angle
        angle = 10000 ** (2*(i//2) / self.d_model)
        self.encoding = pos / angle

        # sin for even dims: 2i
        self.encoding[:, 0::2] = torch.sin(self.encoding[:, 0::2])
        # cos for odd dims: 2i+1
        self.encoding[:, 1::2] = torch.cos(self.encoding[:, 1::2])

    def forward(self):
        return self.encoding
