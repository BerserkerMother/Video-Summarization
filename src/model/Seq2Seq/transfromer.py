import torch
from torch import nn, LongTensor, Tensor
from model.Seq2Seq import PositionalEncoding, MultiHeadAttention


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 in_features,
                 num_class,
                 max_seq_len,
                 device,
                 **configs):
        super().__init__()
        self.d_model = configs['d_model']
        self.dropout_rate = configs['dropout_rate']
        self.max_seq_len = max_seq_len
        self.in_features = in_features
        self.num_class = num_class
        self.device = device

        # input word embeddings
        self._position_enc = PositionalEncoding(
            self.d_model, self.dropout_rate, max_seq_len)

        # input projection
        self._first_layer_enc = nn.Linear(self.in_features, self.d_model)
        self._first_layer_dec = nn.Linear(self.in_features, self.d_model)

        # main parts
        self._encoder = EncoderStack(**configs)
        self._decoder = DecoderStack(**configs)

        # final layer to project decoder output
        self._final_linear = nn.Linear(self.d_model, num_class)

    def forward(self, src: LongTensor, trg: LongTensor):
        src, trg = src, trg

        # positioned_inp_seq = self._position_enc(self._inp_features(src))
        # positioned_trg_seq = self._position_enc(self._trg_word_emb(trg))
        positioned_inp_seq = self._first_layer_enc(src)
        positioned_trg_seq = self._first_layer_dec(src)

        enc_padding_mask, dec_padding_mask, dec_shift_mask = self._create_mask(
            trg, trg)

        enc_out = self._encoder(positioned_inp_seq, enc_padding_mask)
        dec_out = self._decoder(
            positioned_trg_seq, enc_out, dec_padding_mask, dec_shift_mask)
        final_out = self._final_linear(dec_out)

        return final_out

    # ---- Methods --------------------------------------------------------------

    def _padding_mask(self, inp):
        bs, seq_len = inp.size()[0], inp.size()[1]
        return torch.where(inp == 0.0, 1, 0).view(bs, 1, 1, seq_len)

    def _shift_mask(self, inp):
        bs, seq_len = inp.size()[0], inp.size()[1]
        shift_mask = torch.empty(seq_len, seq_len, device=self.device).fill_(
            1).triu(diagonal=1).byte()
        shift_mask = shift_mask.unsqueeze(0).expand(bs, -1, -1)

        return shift_mask.view(bs, 1, seq_len, seq_len)

    def _create_mask(self, input, target):
        # encoder input mask
        enc_padding_mask = self._padding_mask(input)

        # guided att mask in decoder
        dec_gd_padding_mask = self._padding_mask(input)

        # look ahead mask for decoder input
        dec_shift_mask = self._shift_mask(target)
        dec_input_padding_mask = self._padding_mask(target)
        dec_shift_mask = torch.maximum(dec_shift_mask, dec_input_padding_mask)

        return enc_padding_mask, dec_gd_padding_mask, dec_shift_mask


# ============== Encoder =================
class EncoderStack(nn.Module):
    def __init__(self, **configs):
        super().__init__()
        self.num_enc_layer = configs['num_layers']

        self._encoder_stack = nn.ModuleList(
            [EncoderLayer(**configs) for _ in range(self.num_enc_layer)])

    def forward(self, input: Tensor, padding_mask):
        enc_out = input
        for enc in self._encoder_stack:
            enc_out = enc(enc_out, padding_mask)

        return enc_out


class EncoderLayer(nn.Module):
    def __init__(self, **configs):
        super().__init__()
        self.d_model = configs['d_model']
        self.dropout_rate = configs['dropout_rate']

        self._sa = MultiHeadAttention(**configs)

        self._layer_norm = nn.LayerNorm(self.d_model)
        self._layer_norm2 = nn.LayerNorm(self.d_model)

        self._mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model*2, self.d_model)
        )

    def forward(self, x: Tensor, mask=None):
        # self-attention
        sa_out = self._sa(query=x, key=x, value=x, mask=mask)

        # compute residual and get the layer norm
        normalized_att = self._layer_norm(sa_out + x)

        mlp_out = self._mlp(normalized_att)

        normalized_mlp = self._layer_norm2(mlp_out + normalized_att)

        return normalized_mlp


# =============== Decoder ===================
class DecoderStack(nn.Module):
    def __init__(self, **configs):
        super().__init__()
        self.num_dec_layers = configs['num_layers']

        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(**configs) for _ in range(self.num_dec_layers)])

    def forward(self, target: Tensor, enc_out, padding_mask, shift_mask):
        dec_out = target
        for dec in self.decoder_stack:
            dec_out = dec(dec_out, enc_out, padding_mask, shift_mask)

        return dec_out


class DecoderLayer(nn.Module):
    def __init__(self, **configs):
        super().__init__()
        self.d_model = configs['d_model']
        self.dropout_rate = configs['dropout_rate']

        self._sa = MultiHeadAttention(**configs)
        self._guided_att = MultiHeadAttention(**configs)

        self._layer_norm = nn.LayerNorm(self.d_model)
        self._layer_norm2 = nn.LayerNorm(self.d_model)
        self._layer_norm3 = nn.LayerNorm(self.d_model)

        self._mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model*2, self.d_model)
        )

    def forward(self, x: Tensor, enc_out, padding_mask, shift_mask):
        sa = self._sa(query=x, key=x, value=x, mask=shift_mask)
        normalized_sa = self._layer_norm(sa + x)

        guided_att = self._guided_att(
            query=normalized_sa, key=enc_out, value=enc_out, mask=padding_mask)
        normalized_guided = self._layer_norm2(normalized_sa + guided_att)

        mlp = self._mlp(normalized_guided)
        normalized_mlp = self._layer_norm3(mlp + normalized_guided)

        return normalized_mlp
