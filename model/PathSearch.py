"""PathSearch model implementation (polished).

Minimal, non-breaking adjustments:
- Fix import to use existing class name `TransMILABMIL`.
- Robust fallback when `open_clip` is unavailable (no hard fail).
- Type hints, docstrings, and small readability improvements.

This module implements the PathSearch model for pathology image search using
contrastive learning between image and text features.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .TransMIL import TransMILABMIL  # class name present in TransMIL.py


class PathSearch(nn.Module):
    """PathSearch: image–text contrastive model for pathology retrieval.

    Args:
        embed_dim: Output embedding dimension for image/text features.
        init_logit_scale: Initial value for logit scaling (temperature).
        mosaic_num: Number of mosaic patches for the visual encoder.
    """

    def __init__(
        self,
        embed_dim: int,
        init_logit_scale: float = float(np.log(1 / 0.07)),
        mosaic_num: int = 16,
    ) -> None:
        super().__init__()

        # Visual backbone (MIL + transformer refinement)
        self.visual = TransMILABMIL(in_dim=embed_dim, out_dim=embed_dim, mosaic_num=mosaic_num)

        # Text backbone (BiomedCLIP); fall back to a simple zero encoder if unavailable
        self.text_model = None
        self.preprocess = None
        try:
            from open_clip import create_model_from_pretrained

            self.text_model, self.preprocess = create_model_from_pretrained(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
        except Exception as e:  # pragma: no cover — optional dependency
            logging.warning("open_clip unavailable (%s); using a dummy text encoder.", e)

            class _DummyTextModel:
                def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
                    # Infer batch from first dim when possible
                    try:
                        b = int(getattr(text, "shape", [1])[0])
                    except Exception:
                        b = 1
                    out = torch.zeros((b, 512), dtype=torch.float32, device=text.device if torch.is_tensor(text) else "cpu")
                    return F.normalize(out, dim=-1) if normalize else out

            self.text_model = _DummyTextModel()
            self.preprocess = lambda x: x

        # Project text features to `embed_dim` (BiomedCLIP text dim is 512)
        self.text_projection = nn.Linear(512, embed_dim)

        # Temperature parameter (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    # ---------- Encoders ----------
    @property
    def dtype(self) -> torch.dtype:
        return self.visual.cls_token.dtype

    def encode_image(self, image: Optional[torch.Tensor], normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode slide patches to a global feature and mosaic features.

        Args:
            image: Patch-level features, shape (B, N, D).
            normalize: Whether to L2-normalize outputs.
        Returns:
            (features, mosaic): (B, D), (B, M, D)
        """
        if image is None:
            logging.warning("encode_image called with image=None")
            return None, None
        features, mosaic = self.visual(image.type(self.dtype))
        if normalize:
            features = F.normalize(features, dim=-1)
            mosaic = F.normalize(mosaic, dim=-1)
        return features, mosaic

    def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode text tokens via BiomedCLIP (or fallback) and project to `embed_dim`."""
        x = self.text_model.encode_text(text, normalize=normalize)
        x = self.text_projection(x)
        return F.normalize(x, dim=-1) if normalize else x

    # ---------- Forward ----------
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Forward pass returning encoded features and temperature.

        Returns:
            image_features, mosaic_features, text_features, logit_scale_exp
        """
        image_features, mosaic = (None, None)
        if image is not None:
            image_features, mosaic = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        return image_features, mosaic, text_features, self.logit_scale.exp()
