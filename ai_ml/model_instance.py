import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class InstanceUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        # Bottleneck
        self.bottleneck = DoubleConv(
            256,
            512
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.conv3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.conv2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.conv1 = DoubleConv(
            128,
            64
        )

        # IMPORTANT:
        # 3 output channels
        #
        # 0 = Background
        # 1 = Interior
        # 2 = Boundary

        self.final = nn.Conv2d(
            64,
            3,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder

        x1 = self.down1(x)

        x2 = self.pool(x1)
        x2 = self.down2(x2)

        x3 = self.pool(x2)
        x3 = self.down3(x3)

        # Bottleneck

        x4 = self.pool(x3)
        x4 = self.bottleneck(x4)

        # Decoder

        x = self.up3(x4)

        x = torch.cat(
            [x3, x],
            dim=1
        )

        x = self.conv3(x)

        x = self.up2(x)

        x = torch.cat(
            [x2, x],
            dim=1
        )

        x = self.conv2(x)

        x = self.up1(x)

        x = torch.cat(
            [x1, x],
            dim=1
        )

        x = self.conv1(x)

        return self.final(x)