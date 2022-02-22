import logging
import math
from typing import List

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SimNet(nn.Module):

    def __init__(self, num_heads, d_model, num_layers, sparsity, use_cls, dropout,
                 num_classes, use_pos, max_len=2500):
        super(SimNet, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.sparsity = sparsity
        self.use_cls = use_cls
        self.max_len = max_len
        self.num_classes = num_classes
        self.in_features = 1024

        self.embedding_layer = Embedding(
            in_features=self.in_features, d_model=self.d_model,
            use_pos=use_pos, sparsity=sparsity, use_cls=use_cls)

        # importance embeddings
        self.im_emd = nn.Parameter(torch.zeros((1, 10, d_model)))
        self.encoder = Encoder(num_heads, self.d_model, self.num_layers, dropout)
        self.fcs = FullyConnected(
            dim_in=d_model, final_dim=d_model,
            hidden_layers=[d_model * 2, d_model, d_model // 2,
                           d_model, d_model * 2],
            dropout_p=0.4)
        self.im_attention = MultiAttentionNetwork(d_model, d_model,
                                                  num_heads, dropout)
        self.final_layer = nn.Linear(self.d_model, num_classes)

    def forward(self, x, mask=None, vis_attention=False):
        bs, n, _ = x.size()
        x = self.embedding_layer(x)

        # preprocess padding mask
        if isinstance(mask, Tensor):
            mask = self.process_mask(mask)
        # save attention maps
        attention_maps = []
        if vis_attention:
            out = self.encoder(x, mask, attention_maps)
            im_emd = self.im_emd.expand(bs, 10, self.d_model)
            out, _ = self.im_attention(out, im_emd)
            final_out = self.final_layer(out)
            return final_out, attention_maps
        else:
            out = self.encoder(x, mask)
            im_emd = self.im_emd.expand(bs, 10, self.d_model)
            out, _ = self.im_attention(out, im_emd)
            final_out = self.final_layer(out)
            return final_out

    def process_mask(self, mask):
        if self.use_cls:
            cls_mask = torch.tensor([[False]], device=torch.device("cuda"))
            cls_mask = cls_mask.expand(mask.size()[0], 1)
            mask = torch.cat([cls_mask, mask], dim=1)
        batch_size, N = mask.size()
        mask = mask.view(batch_size, 1, 1, N)
        mask = mask.expand(batch_size, self.num_heads, N, N)

        return mask


class Encoder(nn.Module):
    def __init__(self, num_heads, d_model, enc_layers, dropout):
        super(Encoder, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.enc_layers = enc_layers

        modules = []
        for _ in range(self.enc_layers):
            modules.append(
                EncoderBlock(num_heads=self.num_heads, d_model=self.d_model, drop_rate=dropout))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x: Tensor, mask=None, attention_maps=None):
        for block in self.module_list:
            x = block(x, mask, attention_maps)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_heads, d_model, drop_rate=0.3):
        super(EncoderBlock, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.drop_rate = drop_rate

        self.sa = MultiAttentionNetwork(d_model=d_model,
                                        attention_dim=d_model,
                                        num_heads=num_heads,
                                        dropout=drop_rate)
        self.mlp = MLP(d_model=d_model, scale=4, dropout=drop_rate)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=drop_rate)
        self.dropout2 = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor, mask, attention_maps):
        x1, attn = self.sa(x, x, mask=mask)
        x = self.norm1(self.dropout1(x1) + x)  # residual

        x1 = self.mlp(x)
        x = self.norm2(self.dropout2(x1) + x)  # residual

        if isinstance(attention_maps, list):
            attention_maps.append(attn)
        return x


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

    def forward(self, x, y, mask=None):
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
        return attention_output, attention_weight.detach().cpu()


class MLP(nn.Module):
    def __init__(self, d_model, scale=4, dropout=0.2):
        super(MLP, self).__init__()
        # module parameters
        self.d_model = d_model
        self.scale = scale

        # module layers
        self.fc1 = nn.Linear(d_model, scale * d_model)
        self.fc2 = nn.Linear(scale * d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
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


# test for to see how good fully connected layers work
class FullyConnected(nn.Module):
    def __init__(self, dim_in: int, final_dim: int, hidden_layers: List,
                 dropout_p: float):
        super(FullyConnected, self).__init__()
        # model info
        self.dim_in = dim_in
        self.final_dim = final_dim

        # model layers
        layers = [
            nn.Linear(dim_in, hidden_layers[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        ]
        for i in range(len(hidden_layers) - 1):
            layer = nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            activation = nn.ReLU(inplace=True)
            dropout = nn.Dropout(p=dropout_p)
            layers += [layer, activation, dropout]
        # last layer
        layer = nn.Linear(hidden_layers[-1], final_dim)
        activation = nn.ReLU(inplace=True)
        dropout = nn.Dropout(p=dropout_p)
        layers += [layer, activation, dropout]
        # nn.Sequential
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
