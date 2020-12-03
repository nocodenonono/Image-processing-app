import torch.nn as nn
import torch.nn.functional as F
import torch


class ImageTransformNet(nn.Module):
    """
    This implements the image transformation networks
    in this paper: https://arxiv.org/pdf/1603.08155.pdf
    (Perceptual Losses for Real-Time Style Transfer
    and Super-Resolution)

    Exact architecture is given in the supplementary material at Table 1:
    https://web.eecs.umich.edu/~justincj/papers/eccv16/JohnsonECCV16Supplementary.pdf

    Modifications: 1. Pad after each convolution layer
                  2. output raw convolution layer rather than tanh
                  3. Batch normalization => Instance Normalization
    """
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        # Downsampling
        self.conv1 = Conv(3, 32, 9, 1)
        self.conv2 = Conv(32, 64, 3, 2)
        self.conv3 = Conv(64, 128, 3, 2)
        # Residual blocks
        self.res1 = ResNet(128)
        self.res2 = ResNet(128)
        self.res3 = ResNet(128)
        self.res4 = ResNet(128)
        self.res5 = ResNet(128)
        # Upsampling
        self.conv4 = DeConv(128, 64, 3, 1)
        self.conv5 = DeConv(64, 32, 3, 1)
        self.conv6 = Conv(32, 3, 9, 1, False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        out = self.conv6(x)
        return out


class ResNet(nn.Module):
    """
    Residual Network
    """
    def __init__(self, nchannel):
        super(ResNet, self).__init__()
        self.conv1 = Conv(nchannel, nchannel, 3)
        self.conv2 = Conv(nchannel, nchannel, 3)

    def forward(self, x):
        res = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return out + res


class Conv(nn.Module):
    """
    Convolution with 'same' padding.
    the input size is the same as the output size after convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.normalization = nn.InstanceNorm2d(out_channels, affine=True)
        self.norm = norm

    def forward(self, x):
        if self.norm:
            return self.normalization(self.conv(self.pad(x)))
        else:
            return self.conv(self.pad(x))


class DeConv(nn.Module):
    """
    Fractionally convolution layer with normalization and 'same' padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, output_padding=1, padding=1)
        self.normalization = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        return self.normalization(self.conv(x))