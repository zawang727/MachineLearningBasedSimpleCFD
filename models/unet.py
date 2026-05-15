from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CFDUNet(nn.Module):
    """
    Lightweight U-Net surrogate for 2-D steady-state flow prediction.

    Input  (B, in_ch, ny, nx):
      Default (in_ch=5, mesh-aware):
        Ch 0  obstacle mask       (1 = solid, 0 = fluid)
        Ch 1  inlet BC map        (u_inlet where inlet face, else 0)
        Ch 2  lid / wall BC map   (lid_u on lid row, else 0)
        Ch 3  dx / Lx             (per-cell width normalised by domain length)
        Ch 4  dy / Ly

    Output (B, 3, ny, nx):
      Ch 0  u_cell  (cell-centre x-velocity)
      Ch 1  v_cell  (cell-centre y-velocity)
      Ch 2  p       (pressure)
    """

    def __init__(self, base_ch: int = 16, in_channels: int = 5) -> None:
        super().__init__()
        c = base_ch
        self.in_channels = in_channels

        # Encoder
        self.enc1 = _ConvBlock(in_channels, c)
        self.enc2 = _ConvBlock(c,           c * 2)
        self.enc3 = _ConvBlock(c * 2,       c * 4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot = _ConvBlock(c * 4, c * 8)

        # Decoder
        self.up3    = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec3   = _ConvBlock(c*8, c*4)   # concat with enc3 skip

        self.up2    = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2   = _ConvBlock(c*4, c*2)   # concat with enc2 skip

        self.up1    = nn.ConvTranspose2d(c*2, c,   2, stride=2)
        self.dec1   = _ConvBlock(c*2, c)     # concat with enc1 skip

        self.head   = nn.Conv2d(c, 3, 1)     # linear output head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b  = self.bot(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


def build_model(base_ch: int = 16, in_channels: int = 5) -> CFDUNet:
    return CFDUNet(base_ch, in_channels)


def save_model(model: CFDUNet, path: str) -> None:
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path: str, base_ch: int = 16, in_channels: int = 5) -> CFDUNet:
    model = CFDUNet(base_ch, in_channels)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    print(f"Model loaded from {path}")
    return model
