from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CFDUNet3D(nn.Module):
    """
    Lightweight 3-D U-Net surrogate for steady-state flow prediction.

    Input  (B, in_ch, nz, ny, nx):
      Default (in_ch=6, mesh-aware):
        Ch 0  obstacle mask       (1 = solid, 0 = fluid)
        Ch 1  inlet BC map        (u_inlet where inlet face, else 0)
        Ch 2  lid / wall BC map   (lid_u on lid layer, else 0)
        Ch 3  dx / Lx             (per-cell width normalised)
        Ch 4  dy / Ly
        Ch 5  dz / Lz

    Output (B, 4, nz, ny, nx):
      Ch 0  u_cell  (cell-centre x-velocity)
      Ch 1  v_cell  (cell-centre y-velocity)
      Ch 2  w_cell  (cell-centre z-velocity)
      Ch 3  p       (pressure)

    base_ch=8  → ~200K params  (suitable for 32^3 grids)
    base_ch=16 → ~1.5M params  (suitable for 64^3 grids, needs ≥4 GB RAM)
    """

    def __init__(self, base_ch: int = 8, in_channels: int = 6) -> None:
        super().__init__()
        c = base_ch
        self.in_channels = in_channels

        self.enc1 = _ConvBlock3D(in_channels, c)
        self.enc2 = _ConvBlock3D(c,    c * 2)
        self.pool = nn.MaxPool3d(2)
        self.bot  = _ConvBlock3D(c * 2, c * 4)

        self.up2  = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _ConvBlock3D(c * 4, c * 2)   # concat with enc2

        self.up1  = nn.ConvTranspose3d(c * 2, c,     2, stride=2)
        self.dec1 = _ConvBlock3D(c * 2, c)        # concat with enc1

        self.head = nn.Conv3d(c, 4, 1)            # linear output head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                          # (B, c,   nz,   ny,   nx)
        e2 = self.enc2(self.pool(e1))              # (B, 2c,  nz/2, ny/2, nx/2)
        b  = self.bot(self.pool(e2))               # (B, 4c,  nz/4, ny/4, nx/4)

        d2 = self.dec2(torch.cat([self.up2(b),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_model_3d(base_ch: int = 8, in_channels: int = 6) -> CFDUNet3D:
    return CFDUNet3D(base_ch, in_channels)


def save_model_3d(model: CFDUNet3D, path: str) -> None:
    torch.save(model.state_dict(), path)
    print(f"3D model saved to {path}")


def load_model_3d(path: str, base_ch: int = 8, in_channels: int = 6) -> CFDUNet3D:
    model = CFDUNet3D(base_ch, in_channels)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    print(f"3D model loaded from {path}")
    return model
