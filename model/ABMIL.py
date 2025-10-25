"""ABMIL model implementation (polished, minimal edits).

No logic changes; tidy docstrings, type hints, and comments for clarity.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class ResidualBlock(nn.Module):
    """Residual block with two FC layers and ReLU."""

    def __init__(self, n_channels: int = 512) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(n_channels, n_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels, n_channels, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


class DimReduction(nn.Module):
    """Dimensionality reduction with BN+ReLU and optional residuals."""

    def __init__(self, n_channels: int, m_dim: int = 768, num_residual_layers: int = 0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.norm_act = nn.Sequential(
            Rearrange("b n c -> b c n"),
            nn.BatchNorm1d(m_dim),
            Rearrange("b c n -> b n c"),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(m_dim) for _ in range(num_residual_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_act(self.fc1(x))
        for block in self.res_blocks:
            x = block(x)
        return x


class GatedAttention(nn.Module):
    """Gated attention from "Attention-based Deep Multiple Instance Learning" (Ilse et al.)."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, num_heads: int = 1) -> None:
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, num_heads)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        v = self.attention_V(x)
        u = self.attention_U(x)
        a = self.attention_weights(v * u).squeeze(-1)
        return F.softmax(a, dim=1) if normalize else a


class ABMIL(nn.Module):
    """Attention-based MIL encoder producing global and mosaic features.

    Args:
        in_dim: Input feature dim per instance.
        out_dim: Output feature dim.
        mosaic_num: Number of mosaic proxies to compose via attention.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 768, mosaic_num: int = 36) -> None:
        super().__init__()
        self.mosaic_num = mosaic_num
        self.dim_reduction = DimReduction(n_channels=in_dim, m_dim=out_dim)
        self.attention = nn.ModuleList([GatedAttention(input_dim=out_dim, num_heads=1) for _ in range(mosaic_num)])
        self.attention_2 = GatedAttention(input_dim=out_dim, num_heads=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dim_reduction(x)  # (B, N, D)
        b, n, d = x.shape

        mosaics = []
        for att in self.attention:
            w = att(x)  # (B, N)
            mosaic = torch.einsum("bnd,bn->bd", x, w)  # weighted sum over instances
            mosaics.append(mosaic.unsqueeze(1))
        mosaics = torch.cat(mosaics, dim=1)  # (B, M, D)

        w2 = self.attention_2(mosaics)  # (B, M)
        global_feat = torch.einsum("bmd,bm->bd", mosaics, w2)
        return global_feat, mosaics


if __name__ == "__main__":  # basic smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.randn(2, 6000, 512).to(device)
    model = ABMIL(in_dim=512, out_dim=768, mosaic_num=36).to(device)
    model.eval()
    with torch.no_grad():
        f, m = model(data)
        print("Global:", f.shape, "Mosaic:", m.shape)
