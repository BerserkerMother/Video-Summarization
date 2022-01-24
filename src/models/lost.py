
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TLOST(nn.Module):
    """
    Transformer-with-LOcal-attention-and-SumToken!! (TLOST)
    """

    def __init__(self, heads, d_model, num_sumtokens, layers, mask_size, max_len, device):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.num_sumtokens = num_sumtokens
        self.layers = layers
        self.max_len = max_len
        self.mask_size = mask_size
        self.in_features = 1024
        self.device = device

        self.pos_enc = PositionalEncoding(
            self.max_len, self.d_model)

        self.sum_tokens = nn.Embedding(self.num_sumtokens, self.d_model)

        self.first_layer = nn.Linear(self.in_features, self.d_model)

        self.encoder = Encoder(heads, self.d_model, self.layers)
        self.decoder = Decoder(heads, self.d_model, self.layers)

        self.final_layer = nn.Linear(self.d_model, 1)

    def criterian(self, pred, true):
        return F.mse_loss(pred, true)

    def create_local_mask(self, n, size):
        mask1 = torch.ones(n, n).triu(diagonal=size)
        mask2 = torch.ones(n, n).tril(diagonal=-size)
        mask = mask1 + mask2
        idx = torch.tensor(
            [0, int(0.2*n), int(0.4*n), int(0.6*n), int(0.8*n), n-1])
        mask.index_fill_(0, idx, 0)
        mask.index_fill_(1, idx, 0)
        return mask.type(torch.bool)

    def forward(self, x):
        x = self.first_layer(x)
        pe_x = x + self.pos_enc()[:x.size(1), :]

        bs, n, _ = x.size()
        assert self.mask_size < x.size(1)
        local_mask = self.create_local_mask(n, self.mask_size).to(self.device)
        mem = self.encoder(pe_x, local_mask)

        num_shared_toks = int(n / self.num_sumtokens)
        sum_toks = torch.empty(bs, n, self.d_model, device=self.device)
        st = 0
        ed = num_shared_toks
        for i in range(self.num_sumtokens):
            tok = self.sum_tokens(torch.tensor([i], device=self.device))

            if i == self.num_sumtokens - 1:
                last_len = n - ed + num_shared_toks
                tok = tok.view(1, 1, self.d_model).expand(
                    bs, last_len, self.d_model)
                sum_toks[:, -last_len:n, :] = tok
                break

            tok = tok.view(1, 1, self.d_model).expand(
                bs, num_shared_toks, self.d_model)
            sum_toks[:, st:ed, :] = tok
            st = ed
            ed += num_shared_toks

        out = self.decoder(sum_toks, mem)

        final_out = self.final_layer(out)

        return final_out


class Encoder(nn.Module):
    def __init__(self, heads, d_model, enc_layers):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.enc_layers = enc_layers

        modules = []
        for _ in range(self.enc_layers):
            modules.append(
                EncoderBlock(heads=self.heads, d_model=self.d_model, drop_rate=0.3))
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

        self.sa = MHA(heads, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.d_model*2, self.d_model)
        )

    def forward(self, x: Tensor, local_mask):
        attented_x, w = self.sa(q=x, k=x, v=x, mask=local_mask)
        z = self.norm1(attented_x + x)  # residual

        mlp_out = self.mlp(z)
        z2 = self.norm2(mlp_out + z)  # residual

        return z2


class Decoder(nn.Module):
    def __init__(self, heads, d_model, dec_layers):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.dec_layers = dec_layers

        modules = []
        for _ in range(self.dec_layers):
            modules.append(
                DecoderBlock(heads=self.heads, d_model=self.d_model, drop_rate=0.3))
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

        self.sa = MHA(heads, d_model)
        self.ca = MHA(heads, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.d_model*2, self.d_model)
        )

    def forward(self, x: Tensor, mem: Tensor):
        attented_x, w = self.sa(q=x, k=x, v=x, mask=None)
        z = self.norm1(attented_x + x)  # residual

        ca_x, w = self.ca(q=attented_x, k=mem, v=mem, mask=None)
        z2 = self.norm2(ca_x + z)  # residual

        mlp_out = self.mlp(z2)
        z3 = self.norm3(mlp_out + z)  # residual

        return z3


class MHA(nn.Module):
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


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        # pos is the position
        pos = torch.arange(max_len).unsqueeze(dim=1)
        # i is the dimension
        i = torch.arange(d_model)
        self.d_model = d_model
        # for each dimension of d_model compute angle
        angle = 10000 ** (2*(i/2) / self.d_model)
        encoding = pos / angle

        # sin for even dims: 2i
        encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
        # cos for odd dims: 2i+1
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])

        self.register_buffer('encoding', encoding)

    def forward(self):
        return self.encoding
