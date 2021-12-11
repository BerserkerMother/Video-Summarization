import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import process_mask


# Simple transformer network

class MyNetwork(nn.Module):
    def __init__(self, in_features, num_class=5, d_model=256, attention_dim=256,
                 scale=2, num_heads=4, num_layer=3, dropout=0.2):
        super(MyNetwork, self).__init__()

        self.d_model = d_model
        self.num_class = num_class
        self.num_layer = num_layer
        self.num_heads = num_heads

        self.feature_embedding = nn.Linear(in_features, d_model)

        encoder_layer = []
        for _ in range(num_layer):
            encoder_layer.append(TransformerEncoderLayer(d_model, attention_dim, scale, num_heads, dropout))
        self.encoder_layer = nn.ModuleList(encoder_layer)

        self.decoder = nn.Linear(d_model, num_class)

    def forward(self, x, mask):
        x = self.feature_embedding(x)
        mask = process_mask(mask, self.num_heads)
        for module in self.encoder_layer:
            x = module(x, mask)

        x = self.decoder(x)
        # logits = torch.sigmoid(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, attention_dim, scale=4, num_heads=8, dropout=0.2):
        super(TransformerEncoderLayer, self).__init__()

        self.d_model = d_model
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.self_attention = SelfAttentionNetwork(d_model, attention_dim, num_heads, dropout)
        self.mlp = MLP(d_model, scale)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
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

        self.d_model = d_model
        self.attention_dim = attention_dim
        assert attention_dim % num_heads == 0
        self.head_dim = self.attention_dim // num_heads
        self.num_heads = num_heads
        self.scale = d_model ** -0.5

        self.q = nn.Linear(d_model, attention_dim)
        self.k = nn.Linear(d_model, attention_dim)
        self.v = nn.Linear(d_model, attention_dim)

        self.feature_projection = nn.Linear(attention_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        """

        :param x: input dimension (batch_size, N, d_model)
        :return:
        """
        batch_size, N, _ = x.size()

        q = self.q(x).view(batch_size, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).view(batch_size, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).view(batch_size, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attention_score = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attention_score.masked_fill_(mask, float('-inf'))
        attention_weight = F.softmax(attention_score, dim=3)
        attention_weight = self.dropout(attention_weight)

        attention_output = torch.matmul(attention_weight, v).permute(0, 2, 1, 3).contiguous().view(batch_size, N, -1)
        attention_output = self.feature_projection(attention_output)

        return attention_output


class MLP(nn.Module):
    def __init__(self, d_model, scale=4, dropout=0.2):
        super(MLP, self).__init__()

        self.d_model = d_model
        self.scale = scale
        self.fc1 = nn.Linear(d_model, scale * d_model)
        self.fc2 = nn.Linear(scale * d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)

        return x
