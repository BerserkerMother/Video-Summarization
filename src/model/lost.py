import logging
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model import Embedding


class TLOST(nn.Module):
    """
    Transformer-with-LOcal-attention-and-SumToken!! (TLOST)
    """

    def __init__(self, heads, d_model, num_sumtokens, layers, dropout, max_len, device):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.num_sumtokens = num_sumtokens
        self.layers = layers
        self.max_len = max_len
        self.in_features = 1024
        self.vid_len = 320
        self.device = device

        self.max_vid_len = 1294  # this is the max-len in tvsum
        self.patch_size = 30

        emb_size = (self.max_vid_len // self.patch_size) + 1
        self.sum_tokens = nn.Embedding(
            emb_size, self.d_model, device=self.device)

        self.embedding_layer = Embedding(
            in_features=self.in_features, d_model=self.d_model,
            use_pos=True, sparsity=0.0, use_cls=False, use_patch=True, patch_size=self.patch_size)

        self.encoder = Encoder(heads, self.d_model, self.layers, dropout)
        self.decoder = Decoder(heads, self.d_model, self.layers, dropout)

        self.final_layer = nn.Linear(self.d_model, 1)

    def criterian(self, pred, true):
        return F.mse_loss(pred, true)

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
        mem = self.encoder(x, None)
        # sum tokens
        if n % self.patch_size == 0:
            num_patches = (n // self.patch_size)
        else:
            num_patches = (n // self.patch_size) + 1
        sum_tok = self.sum_tokens(torch.tensor(
            [i for i in range(num_patches)], device=self.device))
        out = self.decoder(sum_tok.unsqueeze(0), mem)
        # summ = self.deconv(out)
        summary = self.upsample(out + mem, num_patches, n, bs)

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

        self.sa = MultiAttentionNetwork(d_model=d_model,
                                        attention_dim=d_model,
                                        num_heads=heads,
                                        dropout=drop_rate)
        self.mlp = MLP(d_model=d_model, scale=4, dropout=drop_rate)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=drop_rate)
        self.dropout2 = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor, local_mask):
        attented_x = self.sa(x, x, mask=local_mask)
        z = self.norm1(self.dropout1(attented_x) + x)  # residual

        mlp_out = self.mlp(z)
        z2 = self.norm2(self.dropout2(mlp_out) + z)  # residual

        return z2


class Decoder(nn.Module):
    def __init__(self, heads, d_model, dec_layers, dropout):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.dec_layers = dec_layers

        modules = []
        for _ in range(self.dec_layers):
            modules.append(
                DecoderBlock(heads=self.heads, d_model=self.d_model, drop_rate=dropout))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x: Tensor, mem):
        for block in self.module_list:
            x = block(x, mem)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, heads, d_model, drop_rate=0.3):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.drop_rate = drop_rate

        self.sa = MultiAttentionNetwork(d_model=d_model,
                                        attention_dim=d_model,
                                        num_heads=heads,
                                        dropout=drop_rate)

        self.qa = MultiAttentionNetwork(d_model=d_model,
                                        attention_dim=d_model,
                                        num_heads=heads,
                                        dropout=drop_rate)
        self.mlp = MLP(d_model=d_model, scale=4, dropout=drop_rate)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=drop_rate)
        self.dropout2 = nn.Dropout(p=drop_rate)
        self.dropout3 = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor, mem: Tensor):
        attented_x = self.sa(x, x, mask=None)
        z = self.norm1(self.dropout1(attented_x) + x)  # residual

        ca_x = self.qa(z, mem, mask=None)
        z2 = self.norm2(self.dropout2(ca_x) + z)  # residual

        mlp_out = self.mlp(z2)
        z3 = self.norm3(self.dropout3(mlp_out) + z2)  # residual

        return z3


class MultiAttentionNetwork(nn.Module):
    def __init__(self, d_model, attention_dim, num_heads=8, dropout=0.2):
        super(MultiAttentionNetwork, self).__init__()
        # module parameters
        self.d_model = d_model
        self.attention_dim = attention_dim
        assert attention_dim % num_heads == 0
        self.head_dim = self.attention_dim // num_heads
        self.num_heads = num_heads
        self.scale = d_model ** -0.5

        # module layers
        # q k v layers
        self.q = nn.Linear(d_model, attention_dim)
        self.k = nn.Linear(d_model, attention_dim)
        self.v = nn.Linear(d_model, attention_dim)
        self.dropout = nn.Dropout(p=dropout)

        # self attention projection layer
        self.feature_projection = nn.Linear(attention_dim, d_model)

    def forward(self, x, y, mask):
        """
        :param x: query seq (batch_size, N, d_model)
        :param y: key, value seq
        :param mask: attention mask
        :return:
        """
        batch_size, N, _ = x.size()
        _, M, _ = y.size()

        q = self.q(x).view(batch_size, N, self.num_heads, -1) \
            .permute(0, 2, 1, 3)
        k = self.k(y).view(batch_size, M, self.num_heads, -1) \
            .permute(0, 2, 1, 3)
        v = self.v(y).view(batch_size, M, self.num_heads, -1) \
            .permute(0, 2, 1, 3)

        attention_score = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if isinstance(mask, Tensor):
            attention_score = attention_score.masked_fill(mask, float("-inf"))
        attention_weight = F.softmax(attention_score, dim=3)
        attention_weight = self.dropout(attention_weight)
        attention_output = torch.matmul(attention_weight, v). \
            permute(0, 2, 1, 3).contiguous().view(batch_size, N, -1)

        attention_output = self.feature_projection(attention_output)
        return attention_output


class MLP(nn.Module):
    def __init__(self, d_model, scale=4, dropout=0.2):
        super(MLP, self).__init__()
        # module parameters
        self.d_model = d_model
        self.scale = scale

        # module layers
        self.fc1 = nn.Linear(d_model, scale * d_model)
        self.fc2 = nn.Linear(scale * d_model, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
