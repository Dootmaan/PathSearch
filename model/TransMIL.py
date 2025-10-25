"""TransMIL/ABMIL hybrid encoders (polished).

Minimal, non-breaking adjustments:
- Keep class names `ABMILTransMIL` and `TransMILABMIL` (as in your file) and clean docstrings.
- Consistent type hints & comments; no logic changes.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention

from .ABMIL import ABMIL


class TransLayer(nn.Module):
    """Transformer layer with Nystrom attention and pre-norm."""

    def __init__(self, dim: int = 512, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attn(self.norm(x))
        return x + attended


class PPEG(nn.Module):
    """Pyramid positional encoding generator using depthwise convs."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, n, c = x.shape
        cls, tokens = x[:, 0], x[:, 1:]
        cnn_feat = tokens.transpose(1, 2).view(b, c, height, width)
        pos = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        pos = pos.flatten(2).transpose(1, 2)
        return torch.cat((cls.unsqueeze(1), pos), dim=1)


class ABMILTransMIL(nn.Module):
    """ABMIL → transformer refinement pipeline.

    Returns global feature (CLS) and refined mosaic tokens.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 768, mosaic_num: int = 64) -> None:
        super().__init__()
        self.abmil = ABMIL(in_dim, out_dim, mosaic_num)
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))
        self.trans_layer1 = TransLayer(dim=out_dim)
        self.pos_layer = PPEG(dim=out_dim)
        self.trans_layer2 = TransLayer(dim=out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        semantic, mosaic = self.abmil(x)
        mosaic = self.proj(mosaic)
        num_patches = mosaic.shape[1]
        grid = int(np.ceil(np.sqrt(num_patches)))
        pad = grid * grid - num_patches
        mosaic_pad = torch.cat([mosaic, mosaic[:, :pad, :]], dim=1)
        cls = self.cls_token.expand(mosaic_pad.shape[0], -1, -1)
        x = torch.cat((cls, mosaic_pad), dim=1)
        x = self.trans_layer1(x)
        x = self.pos_layer(x, grid, grid)
        x = self.trans_layer2(x)
        mosaic_refined = x[:, 1 : num_patches + 1, :]
        features = self.norm(x)[:, 0]
        return features, mosaic_refined


class TransMILABMIL(nn.Module):
    """Transformer → ABMIL refinement pipeline.

    Returns global feature and ABMIL-composed mosaic tokens.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 768, mosaic_num: int = 16) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))
        self.trans_layer1 = TransLayer(dim=out_dim)
        self.pos_layer = PPEG(dim=out_dim)
        self.trans_layer2 = TransLayer(dim=out_dim)
        self.abmil = ABMIL(out_dim, out_dim, mosaic_num=mosaic_num)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x.float())
        num_patches = x.shape[1]
        grid = int(np.ceil(np.sqrt(num_patches)))
        pad = grid * grid - num_patches
        x_pad = torch.cat([x, x[:, :pad, :]], dim=1)
        cls = self.cls_token.expand(x_pad.shape[0], -1, -1)
        x = torch.cat((cls, x_pad), dim=1)
        x = self.trans_layer1(x)
        x = self.pos_layer(x, grid, grid)
        x = self.trans_layer2(x)
        x_tokens = x[:, 1 : num_patches + 1, :]
        features, mosaic = self.abmil(x_tokens)
        return features, mosaic


if __name__ == "__main__":  # quick smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 6000, 768).to(device)
    model = TransMILABMIL(in_dim=768, out_dim=768).to(device)
    model.eval()
    with torch.no_grad():
        f, m = model(x)
        print("Global:", f.shape, "Mosaic:", m.shape)
