import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetV4(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()

        self.conv = ConvBlock(in_channels, 64)
        self.enc1 = EncoderBlock(64, 128)
        self.enc2 = EncoderBlock(128, 256)
        self.enc3 = EncoderBlock(256, 512)
        factor = 2 if bilinear else 1
        self.enc4 = EncoderBlock(512, 1024 // factor)
        self.dec1 = DecoderBlock(1024, 512 // factor, bilinear)
        self.dec2 = DecoderBlock(512, 256 // factor, bilinear)
        self.dec3 = DecoderBlock(256, 128 // factor, bilinear)
        self.dec4 = DecoderBlock(128, 64, bilinear)

        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        x6 = self.dec1(x5, x4)
        x6 = self.dec2(x6, x3)
        x6 = self.dec3(x6, x2)
        x6 = self.dec4(x6, x1)
        return self.head(x6)

