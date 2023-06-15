import torch
import torch.nn as nn
from torch.nn import functional as F

import const


class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outChannels, outChannels, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, channels=(5, 16, 32, 64)):
        super().__init__()
        self.encBlocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        for block in self.encBlocks:
            x = block(x)
            x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()

        self.channels = channels
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        encFeatures = nn.CenterCrop([H, W])(encFeatures)
        return encFeatures


class UNetV2(nn.Module):
    def __init__(self, encChannels=(5, 16, 32, 64),
                 decChannels=(64, 32, 16),
                 nbClasses=3, retainDim=True,
                 outSize=const.PIECE_SHAPE):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        self.head = nn.Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])
        map = self.head(decFeatures)
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        return map
