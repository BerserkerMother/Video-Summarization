import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MyNetwork


class PretrainModel(nn.Module):
    """
    implementation of kave's pretrained siamis network
    """

    def __init__(self, feature_dim: int = 128, sparsity: float = 0.7,
                 **kwargs):
        """

        :param feature_dim: dimension of output features
        :param sparsity: sparsity of main encoder
        """
        super(PretrainModel, self).__init__()
        # model parameters
        self.feature_dim = feature_dim
        self.sparsity = sparsity

        # model encoders
        self.encoder_main = MyNetwork(sparsity=sparsity, use_cls=True, **kwargs)
        self.encoder_side = MyNetwork(sparsity=0., use_cls=True, **kwargs)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.encoder_main.parameters(), self.encoder_side.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def forward(self, x, mask):
        batch_size = x.size()[0]

        # forward pass for each network & selects cls token
        sparse_x = self.encoder_main(x, mask)[:, 0]
        with torch.no_grad():
            self._update_momentum_encoder(0.99)

            full_x = self.encoder_side(x, mask)[:, 0]

        # calculate similarities
        full_x = full_x.transpose(0, 1)
        similarities = torch.matmul(sparse_x, full_x)
        # create custom targets, each video must be most similar to it self
        targets = torch.arange(start=0, end=batch_size,
                               device=torch.device("cuda"))

        # calculate loss
        loss = F.cross_entropy(similarities, targets)

        return loss
