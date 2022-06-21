import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .simnet import SimNet


class PretrainModel(nn.Module):
    """
    implementation of pretrained knowledge distillation network
    """

    def __init__(self, feature_dim: int = 256, sparsity: float = 0.0,
                 sharpening_t=0.4, **kwargs):
        """
        :param feature_dim: dimension of output features
        :param sparsity: sparsity of main encoder
        """
        super(PretrainModel, self).__init__()
        # model parameters
        self.feature_dim = feature_dim
        self.sparsity = sparsity
        self.sharpening_t = sharpening_t

        # model encoders
        self.encoder = SimNet(sparsity=0., use_cls=False, d_model=feature_dim,
                              **kwargs)
        # video features transformation
        self.video_transform = nn.Linear(feature_dim, 512)

    def cross_entropy_loss(self, x1, x2):
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)

        loss = -x2 * torch.log(x1)
        # loss = 0.5 * ((x1 - x2) ** 2)
        return loss.mean()

    def entropy(self, x, mask=None):
        x = -(x * torch.log(x))
        if isinstance(mask, Tensor):
            x = x.masked_fill(mask, 0.)
        return x.mean(dim=1).mean()

    def forward(self, x, video_representation, mask=None,
                visualize_attention=None, pen_met="entropy"):

        if visualize_attention:
            out, attention = self.encoder(x, mask, visualize_attention)
        else:
            out = self.encoder(x, mask)
        scores, frame_features = out
        frame_features = self.video_transform(frame_features)

        mask = mask.unsqueeze(2)
        # gets aggregated weights
        if isinstance(mask, Tensor):
            scores = scores.masked_fill(mask, float("-inf"))
        mixture_scores = F.softmax(scores / self.sharpening_t, dim=1)
        # centering loss
        if pen_met == "entropy":
            # adds 1e-9 for numerical stability
            center_loss = self.entropy(mixture_scores + 1e-9, mask)
            print(center_loss)
        else:
            center_loss = torch.norm(mixture_scores, dim=1).mean()
        mixture_scores = mixture_scores.transpose(1, 2)
        video_representation_encoder = torch.matmul(mixture_scores,
                                                    frame_features)
        loss = self.cross_entropy_loss(video_representation_encoder.squeeze(1),
                                       video_representation)
        return loss, center_loss
