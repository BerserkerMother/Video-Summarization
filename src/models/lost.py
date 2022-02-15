import logging
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TLOST(nn.Module):
    """
    Transformer-with-LOcal-attention-and-SumToken!! (TLOST)
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

        self.sum_tokens = nn.Parameter(torch.zeros(self.num_sumtokens, self.d_model))

        self.embedding_layer = Embedding(
            in_features=self.in_features, d_model=self.d_model,
            use_pos=True, sparsity=0.0, use_cls=True)

        self.encoder = Encoder(heads, self.d_model, self.layers, dropout)
        self.decoder = Decoder(heads, self.d_model, self.layers, dropout)

        self.final_layer = nn.Linear(self.d_model, 1)

    def criterian(self, pred, true):
        return F.mse_loss(pred, true)

    def create_local_mask(self, n, size):
        mask1 = torch.ones((n, n), device=self.device).triu(diagonal=size)
        mask2 = torch.ones((n, n), device=self.device).tril(diagonal=-size)
        mask = mask1 + mask2
        idx = torch.tensor(
            [0, int(0.2 * n), int(0.4 * n), int(0.6 * n), int(0.8 * n), n - 1],
            device=self.device)
        mask.index_fill_(0, idx, 0)
        mask.index_fill_(1, idx, 0)
        all_mask = torch.zeros((n + 1, n + 1), device=self.device)
        all_mask[1:, 1:] = mask
        return all_mask.type(torch.bool)

    def forward(self, x):
        bs, n, _ = x.size()
        x = self.embedding_layer(x)

        assert self.mask_size < x.size(1)
        local_mask = self.create_local_mask(n, self.mask_size)
        mem = self.encoder(x, local_mask)
        # sum tokens
        token_expand = (n // self.num_sumtokens) + 1
        sum_toks = self.sum_tokens.view(self.num_sumtokens, 1, -1) \
            .expand(self.num_sumtokens, token_expand, self.d_model)
        sum_toks = sum_toks.contiguous().view(1, -1, self.d_model). \
            expand(bs, self.num_sumtokens * token_expand, self.d_model)
        sum_toks = sum_toks[:, :n, :]
        out = self.decoder(sum_toks, mem)

        # similarity scores
        tokens = mem[:, 1:]
        cls_token = mem[:, 0].unsqueeze(1)
        cos_sim = F.cosine_similarity(tokens, cls_token, dim=2)
        cos_sim = cos_sim / cos_sim.max()

        final_out = torch.sigmoid(self.final_layer(out)).squeeze(-1)

        return final_out * cos_sim


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
