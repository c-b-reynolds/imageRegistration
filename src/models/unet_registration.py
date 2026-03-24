"""
UNet-based registration network with optional diffeomorphic integration.

Architecture overview:
    Encoder: shared-weight conv blocks downsampling both moving & fixed (concatenated)
    Decoder: skip-connection conv blocks upsampling to full resolution
    Head: 1x1 conv predicting velocity/displacement field
    (Optional) VecInt: integrates velocity field -> diffeomorphic displacement
    SpatialTransformer: warps moving image with predicted displacement
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import BaseRegistrationModel
from .spatial_transformer import SpatialTransformer, VecInt


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, ndim: int = 3, stride: int = 1) -> nn.Sequential:
    Conv = nn.Conv3d if ndim == 3 else nn.Conv2d
    return nn.Sequential(
        Conv(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        Conv(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up_block(in_ch: int, out_ch: int, ndim: int = 3) -> nn.Sequential:
    ConvT = nn.ConvTranspose3d if ndim == 3 else nn.ConvTranspose2d
    return nn.Sequential(
        ConvT(in_ch, out_ch, kernel_size=2, stride=2),
        nn.LeakyReLU(0.2, inplace=True),
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class UNetRegistration(BaseRegistrationModel):
    """
    Symmetric UNet for image registration.

    Args:
        image_size:    spatial dimensions of the input (H, W) or (D, H, W)
        in_channels:   channels per image (moving + fixed concatenated = 2 * in_channels)
        base_features: number of feature maps in the first encoder block
        depth:         number of encoder/decoder stages
        int_steps:     VecInt integration steps; 0 disables diffeomorphic integration
        ndim:          2 for 2D images, 3 for 3D volumes
    """

    def __init__(
        self,
        image_size: tuple,
        in_channels: int = 1,
        base_features: int = 16,
        depth: int = 4,
        int_steps: int = 7,
        ndim: int = 3,
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.in_channels = in_channels
        self.base_features = base_features
        self.depth = depth
        self.int_steps = int_steps
        self.ndim = ndim

        # ---- Encoder ----
        enc_chs = [in_channels * 2] + [base_features * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        for i in range(depth):
            stride = 1 if i == 0 else 2
            self.encoders.append(_conv_block(enc_chs[i], enc_chs[i + 1], ndim, stride))

        # ---- Decoder ----
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.upsamples.append(_up_block(enc_chs[i + 1], enc_chs[i], ndim))
            self.decoders.append(_conv_block(enc_chs[i] * 2, enc_chs[i], ndim))

        # ---- Flow head ----
        Conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        self.flow_head = Conv(enc_chs[1], ndim, kernel_size=1)
        # Initialize to near-zero (avoids large initial deformations)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)

        # ---- Integration (diffeomorphic) ----
        self.vec_int: Optional[VecInt] = None
        if int_steps > 0:
            self.vec_int = VecInt(self.image_size, nsteps=int_steps)

        # ---- Spatial transformer ----
        self.transformer = SpatialTransformer(self.image_size)

    # ------------------------------------------------------------------

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([moving, fixed], dim=1)

        # Encoder with skip connections
        skips: List[torch.Tensor] = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < self.depth - 1:
                skips.append(x)

        # Decoder
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # Predict flow / velocity
        flow = self.flow_head(x)

        # Optionally integrate into diffeomorphic displacement
        if self.vec_int is not None:
            flow = self.vec_int(flow)

        warped = self.transformer(moving, flow)

        return {"warped": warped, "flow": flow}

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "base_features": self.base_features,
            "depth": self.depth,
            "int_steps": self.int_steps,
            "ndim": self.ndim,
        }
