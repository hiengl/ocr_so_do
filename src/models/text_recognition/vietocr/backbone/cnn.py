import torch
from torch import nn

from .resnet import Resnet50
from .vgg import Vgg, vgg11_bn, vgg19_bn


class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()

        if backbone == "vgg11_bn":
            self.model = vgg11_bn(**kwargs)
        elif backbone == "vgg19_bn":
            self.model = vgg19_bn(**kwargs)
        elif backbone == "resnet50":
            self.model = Resnet50(**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != "last_conv_1x1":
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
