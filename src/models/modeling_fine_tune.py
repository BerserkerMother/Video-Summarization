from torch import nn
import torch.nn.functional as F
from .modeling_pre_train import Encoder


class EncoderOnly(nn.Module):
    def __init__(self, model, d_enc, heads, enc_layers, mode):
        super().__init__()
        self.model = model
        self.d_enc = d_enc
        self.in_features = 1024
        self.mode = mode

        self.first_layer = nn.Linear(self.in_features, self.d_enc)
        self.encoder = Encoder(heads, d_enc, enc_layers)
        self.regression_out = nn.Linear(self.d_enc, 1)

        self.first_layer.load_state_dict(self.model.first_layer.state_dict())
        self.encoder.load_state_dict(self.model.encoder.state_dict())

        if self.mode == "freeze":
            # freeze the whole model and only update last linear layer
            for param in self.model.parameters():
                param.requires_grad = False

    def fine_tune_criterian(self, pred, target):
        return F.mse_loss(pred, target)

    def forward(self, x, mask_idx, vis_idx):
        enc_out = self.encoder(self.first_layer(x))
        out = self.regression_out(enc_out)
        return out.squeeze(dim=-1)
