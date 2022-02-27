"""this module contains feature extractor models from original pytorch
implementation with pretrain weights"""
import torch
import torch.nn as nn
from torchvision import models


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        # get pretrain model from torchvision
        model = models.googlenet(pretrained=True,
                                 progress=True,
                                 aux_logits=False)
        model_modules = list(model.children())
        # since we don't need the last linear and dropout
        need_modules = model_modules[:-2]
        # create model as nn.Sequential
        self.model = nn.Sequential(*need_modules)

    def forward(self, x):
        """
        Take image tensors and outputs GoogleNet feature

        :param x: image tensors (batch_size, 3, H,W)
        :type x: Tensor
        :returns: image deep features (batch_size, 1024)
        """

        with torch.no_grad():
            x = self.model(x)
            x = torch.flatten(x, 1)
            return x


class R3D18(nn.Module):
    def __init__(self):
        super(R3D18, self).__init__()
        # get pretrain model from torchvision
        model = models.video.r3d_18(pretrained=True,
                                    progress=True)
        model_modules = list(model.children())
        # since we don't need the last linear
        need_modules = model_modules[:-1]
        # create model as nn.Sequential
        self.model = nn.Sequential(*need_modules)

    def forward(self, x):
        """
        Take video tensors and outputs R3D18 feature

        :param x: image tensors (batch_size, 3, T, H, W)
        :type x: Tensor
        :returns: video deep features (batch_size, 1024)
        """

        with torch.no_grad():
            x = self.model(x)
            x = torch.flatten(x, 1)
            return x
