import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MyNetwork


class PretrainModel(nn.Module):
    """
    implementation of kave's pretrained siamis network
    """

    def __init__(self, feature_dim: int = 128, sparsity: float = 0.7,
                 memory_size: int = 8, **kwargs):
        """

        :param feature_dim: dimension of output features
        :param sparsity: sparsity of main encoder
        """
        super(PretrainModel, self).__init__()
        # model parameters
        self.feature_dim = feature_dim
        self.sparsity = sparsity
        self.memory_size = memory_size

        # model encoders
        self.encoder_main = MyNetwork(sparsity=sparsity, use_cls=True, **kwargs)
        self.encoder_side = MyNetwork(sparsity=0., use_cls=True, **kwargs)

        # encoder side memory buffer
        self.register_buffer("memory",
                             torch.randn(feature_dim, memory_size))
        self.memory = F.normalize(self.memory, dim=0)
        self.register_buffer("memory_pointer", torch.zeros(1, dtype=torch.long))

    def queue(self, x):
        batch_size = x.size()[1]

        # covert pointer to int
        ptr = int(self.memory_pointer)

        # replace data
        self.memory[:, ptr: ptr + batch_size] = x

        # update pointer
        ptr = (batch_size + ptr) % self.memory_size
        self.memory_pointer[0] = ptr

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in \
                zip(self.encoder_main.parameters(), self.encoder_side.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def forward(self, x, mask):
        batch_size = x.size()[0]

        # forward pass for each network & selects cls token
        sparse_x = F.normalize(self.encoder_main(x, mask)[:, 0], dim=1)
        with torch.no_grad():
            self._update_momentum_encoder(0.99)

            full_x = F.normalize(self.encoder_side(x, mask)[:, 0], dim=1)

        # calculate similarities
        full_x = full_x.transpose(0, 1)
        similarities_online = torch.matmul(sparse_x, full_x)
        similarities_offline = torch.matmul(sparse_x, self.memory)
        similarities = torch.cat([similarities_online, similarities_offline],
                                 dim=1)
        # create custom targets
        targets = torch.arange(start=0, end=batch_size,
                               device=torch.device("cuda"))

        # calculate loss
        loss = F.cross_entropy(similarities, targets)

        # update queue
        self.queue(full_x)

        return loss
