import torch.nn as nn
import torchvision.models as models


class MyVgg(nn.Module):
    """
    Downloads the pretrained vgg16 model and extract
    layers from it
    """

    def __init__(self):
        super(MyVgg, self).__init__()
        self.features = models.vgg16(pretrained=True).features.to('cuda').eval()
        self.layers = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        feature_maps = []
        for layer, module in self.features._modules.items():
            x = module(x)
            if layer in self.layers:
                feature_maps.append(x)

        return feature_maps
