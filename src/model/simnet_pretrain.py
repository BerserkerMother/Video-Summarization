import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from simnet import SimNet


class PretrainModel(nn.Module):
    """
    implementation of pretrained knowledge distillation network
    """

    def __init__(self, feature_dim: int = 512, sparsity: float = 0.5,
                 sharpening_t=0.9, **kwargs):
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
        self.encoder = SimNet(sparsity=0., use_cls=True, d_model=feature_dim,
                              **kwargs)

    def cross_entropy_loss(self, x1, x2):
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)

        loss = x2 * torch.log(x1)
        return loss.mean() * -1

    def forward(self, x, video_representation, mask=None,
                visualize_attention=None):
        batch_size = x.size()[0]

        if visualize_attention:
            out, attention = self.encoder(x, mask, visualize_attention)
        else:
            out = self.encoder(x, mask)
        scores, frame_features = out

        # center and sharpen scores
        center_vec = scores - torch.mean(scores, dim=0, keepdim=True)
        mixture_scores = F.softmax(center_vec / self.sharpening_t, dim=1)
        mixture_scores = mixture_scores.transpose(1, 2)
        video_representation_encoder = torch.matmul(mixture_scores,
                                                    frame_features)
        loss = self.cross_entropy_loss(video_representation_encoder.squeeze(1),
                                       video_representation)
        print(loss)

        return loss


model = PretrainModel(
    num_classes=1,
)
x1 = torch.randn((2, 26, 1024))
x2 = torch.randn((2, 512))

model(x1, x2)
