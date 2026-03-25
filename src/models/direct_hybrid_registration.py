"""
Direct Hybrid registration.

Uses the same HybridVelocityNet as the ODE-based models (GatedCorrelationStem
+ Transformer + HaarDecoder) but directly predicts a displacement field with
no ODE integration.

The network predicts a displacement u(r) in normalized [-1, 1] coordinates.
The deformation field phi = identity + u is used to warp the moving image:

    warped(r) = moving(phi(r)) = moving(r + u(r))

Output dict:
    warped  (B, 1, H, W)   registered image
    phi     (B, 2, H, W)   deformation field in [-1, 1] coords
                            (compatible with deformation_gradient regularization)
    flow    (B, 2, H, W)   displacement field u = phi - identity
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRegistrationModel
from .hybrid_ode_registration import HybridVelocityNet, _make_identity_grid


class DirectHybridRegistration(BaseRegistrationModel):
    """
    Single-pass displacement regression using the Hybrid velocity network.

    The network predicts a displacement field in one forward pass — no ODE.
    Faster and simpler than the ODE variants; useful as a baseline.

    Constructor arguments are identical to EulerianHybridODERegistration
    (minus ODE-specific params) so the same stem/transformer hyperparameters
    can be compared directly.

    Recommended loss config:
        similarity:   ncc
        regularization: deformation_gradient
        regularization_weight: 0.1
    """

    def __init__(
        self,
        image_size: tuple,
        in_channels: int = 1,
        # Stem
        stem_channels: int = 64,
        kernel_size: int = 8,
        stride: int = 8,
        padding: int = 0,
        pool_type: str = "max",
        stem_hidden_dim: int = 64,
        # Transformer
        embed_dim: int = 256,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        nhead: int = 4,
        ffn_dim: int = None,
        tgt_mode: str = "encoder_output",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size         = tuple(image_size)
        self.in_channels        = in_channels
        self.stem_channels      = stem_channels
        self.kernel_size        = kernel_size
        self.stride             = stride
        self.padding            = padding
        self.pool_type          = pool_type
        self.stem_hidden_dim    = stem_hidden_dim
        self.embed_dim          = embed_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.nhead              = nhead
        self.ffn_dim            = ffn_dim or embed_dim * 4
        self.tgt_mode           = tgt_mode
        self.dropout            = dropout

        self.displacement_net = HybridVelocityNet(
            image_size=self.image_size,
            in_channels=in_channels,
            stem_channels=stem_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool_type=pool_type,
            stem_hidden_dim=stem_hidden_dim,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            nhead=nhead,
            ffn_dim=self.ffn_dim,
            tgt_mode=tgt_mode,
            dropout=dropout,
        )

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = moving.shape

        # Predict displacement in normalized [-1, 1] coords
        u = self.displacement_net(moving, fixed)   # (B, 2, H, W)

        # Deformation field: phi = identity + displacement
        identity = _make_identity_grid(B, H, W, moving.device, moving.dtype)
        phi = identity + u                          # (B, 2, H, W)

        # Warp moving image using phi
        grid   = phi.permute(0, 2, 3, 1)           # (B, H, W, 2)
        warped = F.grid_sample(
            moving, grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )

        return {"warped": warped, "phi": phi, "flow": u}

    def get_config(self) -> Dict[str, Any]:
        return {
            "image_size":          self.image_size,
            "in_channels":         self.in_channels,
            "stem_channels":       self.stem_channels,
            "kernel_size":         self.kernel_size,
            "stride":              self.stride,
            "padding":             self.padding,
            "pool_type":           self.pool_type,
            "stem_hidden_dim":     self.stem_hidden_dim,
            "embed_dim":           self.embed_dim,
            "num_encoder_layers":  self.num_encoder_layers,
            "num_decoder_layers":  self.num_decoder_layers,
            "nhead":               self.nhead,
            "ffn_dim":             self.ffn_dim,
            "tgt_mode":            self.tgt_mode,
            "dropout":             self.dropout,
        }
