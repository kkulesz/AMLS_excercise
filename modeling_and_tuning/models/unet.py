import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.downs = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Decoder
        self.ups = torch.nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, out_channels, kernel_size=1)
        ])

    def forward(self, x):
        for down_layer in self.downs:
            x = down_layer(x)

        x = self.bottleneck_conv(x)

        for up_layer in self.ups:
            x = up_layer(x)

        return x
