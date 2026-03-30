"""
GateFlowMSAttn: Multi-scale GateFlow with spatial self-attention.

Extends GateFlowMS by inserting a multi-head self-attention (MHSA) block
immediately after the bottleneck, before the PE residual branch:

    ... → bottleneck → [MHSA over H'×W' tokens] → PE branch → vel_head → ...

The coarse stem (stride s2) produces feature maps at H/s1 × W/s1 (e.g. 16×16
for a 64×64 input with s1=4), giving 256 tokens — cheap for standard attention.

Motivation
----------
RandomErasing and poor-content regions produce near-zero or noisy gate values
at those spatial locations.  Without attention, the PE branch at an erased
location operates on a degraded gate signal in isolation.  MHSA allows every
spatial location to attend to every other, so well-conditioned locations
(strong gate signal) can propagate context into erased or low-quality regions.

This complements the sech² coarse-inhibition from GateFlowMS: when coarse
suppresses fine features in a region, attention lets the bottleneck borrow
from adjacent locations where fine features remain active.

Position in the forward pass
-----------------------------
Attention is placed AFTER the bottleneck so it operates on the compressed,
nonlinearly-activated feature representation rather than the raw gate product.
No additional positional encoding is injected into the attention — spatial
awareness is handled downstream by the existing PE residual branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Optional

from .gate_flow import GateFlow, _make_pe
from .gate_flow_ms import GateFlowMS, GateFlowMSVelocityNet


class GateFlowMSAttnVelocityNet(GateFlowMSVelocityNet):
    """
    Multi-scale velocity net with post-bottleneck spatial self-attention.

    Extra args:
        n_heads      : number of attention heads (hidden_channels must be
                       divisible by n_heads)
        attn_dropout : dropout probability inside attention (default 0.0)
    """

    def __init__(
        self,
        in_channels:         int,
        hidden_channels:     int,
        kernel_size:         int,
        stride:              int,
        bottleneck_channels: int,
        n_pe:                int,
        kernel_size_2:       int,
        stride_2:            int,
        n_heads:             int  = 8,
        attn_dropout:        float = 0.0,
        image_size:          Optional[List[int]] = None,
        pe_activation:       Optional[str] = None,
        pe_type:             str = "fixed",
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            n_pe=n_pe,
            kernel_size_2=kernel_size_2,
            stride_2=stride_2,
            image_size=image_size,
            pe_activation=pe_activation,
            pe_type=pe_type,
        )
        C = hidden_channels
        if C % n_heads != 0:
            raise ValueError(
                f"hidden_channels ({C}) must be divisible by n_heads ({n_heads})"
            )
        self.attn = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=False,   # expects (seq, batch, embed)
        )

    def forward(self, f: Tensor, g: Tensor) -> Tensor:
        H, W = f.shape[-2], f.shape[-1]

        # --- Fine stem ---
        ff1 = self.conv_f(f)
        gf1 = self.conv_g(g)
        prod_fine = ff1 * gf1
        gate_fine = prod_fine[:, :self.C] + prod_fine[:, self.C:]

        # --- Coarse stem ---
        ff2 = self.conv_f2(f)
        gf2 = self.conv_g2(g)
        prod_coarse = ff2 * gf2
        gate_coarse = prod_coarse[:, :self.C] + prod_coarse[:, self.C:]

        # --- Cross-scale inhibition (sech²) ---
        gate_coarse_up = F.interpolate(gate_coarse, size=gate_fine.shape[-2:],
                                       mode="bilinear", align_corners=True)
        coarse_tanh = torch.tanh(gate_coarse_up)
        gate = gate_fine * (1.0 - coarse_tanh.pow(2))

        # --- Bottleneck ---
        gate = self.expand(torch.tanh(self.squeeze(gate)))  # (N, C, H', W')

        # --- Spatial self-attention over H'×W' tokens ---
        N, C, H_p, W_p = gate.shape
        x = gate.flatten(2).permute(2, 0, 1)   # (H'*W', N, C)
        x, _ = self.attn(x, x, x)              # (H'*W', N, C)
        gate = x.permute(1, 2, 0).view(N, C, H_p, W_p)  # (N, C, H', W')

        # --- Positional encoding ---
        if self.pe_type == "learned":
            pe = self._learned_pe.to(dtype=f.dtype)
        elif (self._pe_buf is not None
              and self._pe_buf.shape[-2] == H_p
              and self._pe_buf.shape[-1] == W_p):
            pe = self._pe_buf.to(dtype=f.dtype)
        else:
            pe = _make_pe(H_p, W_p, self.n_pe, device=f.device, dtype=f.dtype)

        x = self.pe_in(pe)
        x = x * gate
        if self.pe_act is not None:
            x = self.pe_act(x)
        x = self.pe_out(x)
        x = x - pe
        v = self.vel_head(x)

        return F.interpolate(v, size=(H, W), mode="bilinear", align_corners=True)


class GateFlowMSAttn(GateFlowMS):
    """
    Multi-scale GateFlow with spatial self-attention after the bottleneck.

    Extra args (on top of GateFlowMS):
        n_heads      : attention heads  (hidden_channels % n_heads == 0)
        attn_dropout : dropout inside attention
    """

    def __init__(
        self,
        in_channels:         int                 = 1,
        hidden_channels:     int                 = 32,
        kernel_size:         int                 = 8,
        stride:              int                 = 4,
        bottleneck_channels: int                 = 16,
        n_pe:                int                 = 32,
        n_t:                 int                 = 5,
        kernel_size_2:       int                 = 16,
        stride_2:            int                 = 8,
        n_heads:             int                 = 8,
        attn_dropout:        float               = 0.0,
        image_size:          Optional[List[int]] = None,
        pe_activation:       Optional[str]       = None,
        pe_type:             str                 = "fixed",
    ):
        super(GateFlowMS, self).__init__()   # BaseRegistrationModel
        self.n_t = n_t

        self.velocity_net = GateFlowMSAttnVelocityNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            n_pe=n_pe,
            kernel_size_2=kernel_size_2,
            stride_2=stride_2,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            image_size=image_size,
            pe_activation=pe_activation,
            pe_type=pe_type,
        )

        self._init_kwargs: Dict[str, Any] = dict(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            n_pe=n_pe,
            n_t=n_t,
            kernel_size_2=kernel_size_2,
            stride_2=stride_2,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            image_size=list(image_size) if image_size is not None else None,
            pe_activation=pe_activation,
            pe_type=pe_type,
        )

    def parameter_summary(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        vn     = self.velocity_net
        pe_act = vn.pe_act.__class__.__name__ if vn.pe_act is not None else "none"
        return (
            f"GateFlowMSAttn  C={vn.C}  K={vn.squeeze.out_channels}  "
            f"k1={vn.kernel_size}  s1={vn.stride}  "
            f"k2={vn.conv_f2.weight.shape[-1]}  s2={vn.conv_f2.stride}  "
            f"n_pe={vn.n_pe}  n_t={self.n_t}  "
            f"n_heads={vn.attn.num_heads}  "
            f"pe_type={vn.pe_type}  pe_act={pe_act}\n"
            f"Parameters: {total:,} total  {trainable:,} trainable"
        )
