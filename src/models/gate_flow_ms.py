"""
GateFlowMS: Multi-scale extension of GateFlow.

Adds a second coarse-scale stem pair (conv_f2, conv_g2) with a larger kernel
and stride.  The coarse gate modulates the fine gate via a tanh cross-scale
interaction before the shared bottleneck:

    gate_fine   = fold( conv_f1(moving) * conv_g1(fixed) )   # fine resolution
    gate_coarse = fold( conv_f2(moving) * conv_g2(fixed) )   # coarse resolution

    gate = gate_fine * tanh( upsample(gate_coarse) )

This is strictly more interconnected than a two-tower design: the coarse
gate shapes which fine-scale features survive compression, rather than
simply blending two independent velocity outputs.  tanh is used (not sigmoid)
to preserve the signed response of the gating product — the coarse context
can reinforce, suppress, or invert fine-scale matches.

Everything downstream (bottleneck, PE residual, upsample, iterative loop)
is inherited from GateFlow unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .gate_flow import GateFlow, GateFlowVelocityNet, L2NormConv2d, _make_pe


class GateFlowMSVelocityNet(GateFlowVelocityNet):
    """
    Multi-scale velocity net.  Inherits all of GateFlowVelocityNet and adds
    a second (coarse) stem pair whose gate modulates the fine gate before
    the shared bottleneck.

    Extra args:
        kernel_size_2 (k2): square kernel for the coarse stem  (should be > k)
        stride_2      (s2): stride for the coarse stem          (should be > s)
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
            image_size=image_size,
            pe_activation=pe_activation,
            pe_type=pe_type,
        )
        C = hidden_channels
        padding_2 = (kernel_size_2 - stride_2) // 2

        self.conv_f2 = L2NormConv2d(in_channels, 2 * C, kernel_size_2,
                                    stride=stride_2, padding=padding_2)
        self.conv_g2 = L2NormConv2d(in_channels, 2 * C, kernel_size_2,
                                    stride=stride_2, padding=padding_2)

    def forward(self, f: Tensor, g: Tensor) -> Tensor:
        H, W = f.shape[-2], f.shape[-1]

        # --- Fine stem ---
        ff1 = self.conv_f(f)    # (N, 2C, H/s1, W/s1)
        gf1 = self.conv_g(g)
        prod_fine   = ff1 * gf1
        gate_fine   = prod_fine[:, :self.C] + prod_fine[:, self.C:]   # (N, C, H/s1, W/s1)

        # --- Coarse stem ---
        ff2 = self.conv_f2(f)   # (N, 2C, H/s2, W/s2)
        gf2 = self.conv_g2(g)
        prod_coarse = ff2 * gf2
        gate_coarse = prod_coarse[:, :self.C] + prod_coarse[:, self.C:]  # (N, C, H/s2, W/s2)

        # --- Cross-scale modulation (coarse gates fine) ---
        gate_coarse_up = F.interpolate(gate_coarse, size=gate_fine.shape[-2:],
                                       mode="bilinear", align_corners=True)
        gate = gate_fine * torch.tanh(gate_coarse_up)   # (N, C, H/s1, W/s1)

        # --- Bottleneck ---
        gate = self.expand(torch.tanh(self.squeeze(gate)))  # (N, C, H/s1, W/s1)

        # --- Positional encoding ---
        H_p, W_p = gate.shape[-2], gate.shape[-1]
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
        v = self.vel_head(x)    # (N, 2, H/s1, W/s1)

        return F.interpolate(v, size=(H, W), mode="bilinear", align_corners=True)


class GateFlowMS(GateFlow):
    """
    Multi-scale GateFlow.  Identical to GateFlow except the velocity network
    uses a second coarse stem pair to provide long-range spatial context.

    Extra args:
        kernel_size_2: coarse stem kernel size  (recommend 2× kernel_size)
        stride_2:      coarse stem stride       (recommend 2× stride)
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
        image_size:          Optional[List[int]] = None,
        pe_activation:       Optional[str]       = None,
        pe_type:             str                 = "fixed",
    ):
        # Bypass GateFlow.__init__ and call BaseRegistrationModel directly,
        # then build our own velocity_net and _init_kwargs.
        super(GateFlow, self).__init__()
        self.n_t = n_t

        self.velocity_net = GateFlowMSVelocityNet(
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
            f"GateFlowMS  C={vn.C}  K={vn.squeeze.out_channels}  "
            f"k1={vn.kernel_size}  s1={vn.stride}  "
            f"k2={vn.conv_f2.weight.shape[-1]}  s2={vn.conv_f2.stride}  "
            f"n_pe={vn.n_pe}  n_t={self.n_t}  "
            f"pe_type={vn.pe_type}  pe_act={pe_act}\n"
            f"Parameters: {total:,} total  {trainable:,} trainable"
        )
