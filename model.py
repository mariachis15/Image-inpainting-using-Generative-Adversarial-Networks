import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()

        self.d1 = DownBlock(in_channels, 64, norm=False)
        self.d2 = DownBlock(64, 128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512)
        self.d5 = DownBlock(512, 512)
        self.d6 = DownBlock(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.u1 = UpBlock(512, 512, dropout=True)
        self.u2 = UpBlock(1024, 512, dropout=True)
        self.u3 = UpBlock(1024, 512, dropout=True)
        self.u4 = UpBlock(1024, 256)
        self.u5 = UpBlock(512, 128)
        self.u6 = UpBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, masked_image, mask):
        x = torch.cat([masked_image, mask], dim=1)

        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)

        b = self.bottleneck(d6)

        u1 = self.u1(b)
        u1 = torch.cat([u1, d6], dim=1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d5], dim=1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d4], dim=1)

        u4 = self.u4(u3)
        u4 = torch.cat([u4, d3], dim=1)

        u5 = self.u5(u4)
        u5 = torch.cat([u5, d2], dim=1)

        u6 = self.u6(u5)
        u6 = torch.cat([u6, d1], dim=1)

        out = self.final(u6)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        return self.model(x)
