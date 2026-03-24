"""
Minimal CNN registration network.

Architecture:
    [moving | fixed] → Conv → Conv → Conv → Conv → flow (2, H, W)
                                                        ↓
                                              SpatialTransformer
                                                        ↓
                                                    warped

No downsampling, no skip connections — just a stack of conv layers at full
resolution. Effective for small images and quick experimentation.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseRegistrationModel
from .spatial_transformer import SpatialTransformer, VecInt


class SimpleCNN(BaseRegistrationModel):
    """
    Args:
        image_size:   (H, W) spatial size of input images.
        features:     Number of feature maps in the hidden layers.
        depth:        Number of hidden conv layers (excluding the flow head).
        int_steps:    VecInt integration steps; 0 = plain displacement field.
    """

    def __init__(
        self,
        image_size: tuple,
        in_channels: int = 1,
        features: int = 32,
        depth: int = 4,
        int_steps: int = 0,
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.in_channels = in_channels
        self.features   = features
        self.depth      = depth
        self.int_steps  = int_steps

        layers = []
        in_ch  = in_channels * 2    # moving + fixed concatenated along channel dim
        for i in range(depth):
            out_ch = features
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)

        # Flow head — outputs 2-channel displacement field
        self.flow_head = nn.Conv2d(features, 2, kernel_size=1)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)

        self.vec_int     = VecInt(self.image_size, nsteps=int_steps) if int_steps > 0 else None
        self.transformer = SpatialTransformer(self.image_size)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> Dict[str, torch.Tensor]:
        x    = torch.cat([moving, fixed], dim=1)   # (B, 2, H, W)
        x    = self.encoder(x)
        flow = self.flow_head(x)                   # (B, 2, H, W)

        if self.vec_int is not None:
            flow = self.vec_int(flow)

        warped = self.transformer(moving, flow)
        return {"warped": warped, "flow": flow}

    def get_config(self) -> Dict[str, Any]:
        return {
            "image_size":  self.image_size,
            "in_channels": self.in_channels,
            "features":    self.features,
            "depth":       self.depth,
            "int_steps":   self.int_steps,
        }
