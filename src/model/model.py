import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Simple transformer network

class MyNetwork(nn.Module):
    def __init__(self, in_features, num_class=1, d_model=256,
                 attention_dim=256, scale=4, num_heads=4, num_layer=3,
                 dropout=0.2, sparsity=0.7,
                 use_pos: bool = True,
                 use_cls: bool = True):
        super(MyNetwork, self).__init__()
        # module parameters
        self.d_model = d_model
        self.num_class = num_class
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.use_pos = use_pos

        # module layers
        # embedding layer
        self.embedding = Embedding(in_features=in_features, d_model=d_model,
                                   sparsity=sparsity, use_pos=use_pos,
                                   use_cls=use_cls)

        # encoder layers
        encoder_layer = []
        for _ in range(num_layer):
            layer = TransformerEncoderLayer(d_model, attention_dim,
                                            scale, num_heads, dropout)
            encoder_layer.append(layer)
        self.encoder_layer = nn.ModuleList(encoder_layer)

        # decoder MLP
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_class)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        batch_size = x.size()[0]

        if isinstance(mask, Tensor):
            mask = self.process_mask(mask)
        x = self.embedding(x)
        for module in self.encoder_layer:
            x = module(x, mask)

        x = self.dropout(F.relu(self.fc1(self.norm1(x))))
        logits = self.fc2(self.norm2(x))
        return logits

    def process_mask(self, mask: Tensor, use_cls: bool = True):
        """

        :param mask: bool mask if size (batch_size, N)
        :param use_cls: true if using a cls token
        :return: shaped mask of size (batch_size, num_heads, N, N)
        N+1 if using cls token
        """
        batch_size = mask.size()[0]

        if use_cls:
            cls_mask = torch.zeros((batch_size, 1), device=torch.device("cuda"))
            mask = torch.cat([cls_mask, mask], dim=1)

        N = mask.size()[1]
        mask = mask.view(batch_size, 1, 1, N)
        mask = mask.expand(batch_size, self.num_heads, N, N).type(torch.bool)
        return mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, attention_dim, scale=4,
                 num_heads=8, dropout=0.2):
        super(TransformerEncoderLayer, self).__init__()
        # module parameters
        self.d_model = d_model
        self.attention_dim = attention_dim
        self.num_heads = num_heads

        # module layers
        # self attention
        self.self_attention = SelfAttentionNetwork(d_model, attention_dim,
                                                   num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        # mlp
        self.mlp = MLP(d_model, scale)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # self attention
        x = self.dropout1(self.self_attention(self.norm1(x), mask)) + x

        # mlp
        x = self.dropout2(self.mlp(self.norm2(x))) + x
        return x


class SelfAttentionNetwork(nn.Module):
    def __init__(self, d_model, attention_dim, num_heads=8, dropout=0.2):
        super(SelfAttentionNetwork, self).__init__()
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

    def forward(self, x, mask):
        """

        :param x: input dimension (batch_size, N, d_model)
        :return:
        """
        batch_size, N, _ = x.size()

        q = self.q(x).view(batch_size, N, self.num_heads, -1) \
            .permute(0, 2, 1, 3)
        k = self.k(x).view(batch_size, N, self.num_heads, -1) \
            .permute(0, 2, 1, 3)
        v = self.v(x).view(batch_size, N, self.num_heads, -1) \
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
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class Embedding(nn.Module):
    def __init__(self, in_features, d_model: int = 512,
                 max_len: int = 2500, sparsity: float = 0.7,
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
