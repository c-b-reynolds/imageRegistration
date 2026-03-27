"""
GateFlow: Iterative Lagrangian image registration with a gated convolutional
displacement network and positional-encoding residual.

Architecture
------------
1. Stem
   Two separate Conv2d layers (L2-normalised weights, no bias), one per image:
       conv_f, conv_g : (N, 1, H, W) -> (N, 2C, H', W')
   Padding = (k - s) / 2 keeps H' = H / s exactly.

2. Gate
   Element-wise product of the two stem outputs, then fold:
       prod = conv_f(f) * conv_g(g)          # (N, 2C, H', W')
       gate = prod[:, :C] + prod[:, C:]       # (N,  C, H', W')

3. Bottleneck on the gate
       gate = expand( tanh( squeeze(gate) ) )
   where squeeze: Conv1x1 C->K,  expand: Conv1x1 K->C.

4. Positional-encoding residual branch
   A fixed sinusoidal PE encodes the [-1, 1]^2 unit grid at coarse res (H', W').
   n_pe channels (divisible by 4):  n_pe/4 frequencies contribute
       [cos(f*x), sin(f*x), cos(f*y), sin(f*y)]  each.

       pe                  : (1, n, H', W')   fixed, not learned
       pe_in(pe) * gate    : Conv1x1 n->C, element-wise with gate
       pe_out(...)         : Conv1x1 C->n
       delta = result - pe : "change of positional encoding"
       vel_head(delta)     : Conv1x1 n->2  → incremental displacement

5. Upsample
   Bilinear interpolation from (H', W') back to (H, W).

6. Iterative Lagrangian integration  (n_t steps)
   phi_0  = 0
   for i in 1..n_t:
       delta_i  = velocity_net( warp(moving, phi_{i-1}), fixed )
       phi_i    = phi_{i-1} + delta_i          # additive — valid for small deltas
   warped = warp(moving, phi_{n_t})

   Correctness of addition vs. composition:
   For infinitesimal delta_i,  phi_j(r + phi_i(r)) ≈ phi_j(r)  (first-order),
   so sum ≈ composition.  The network is regularised to keep each delta small.

Loss compatibility
------------------
  'flow' / 'phi'  — total accumulated displacement (normalised coords).
  Use  regularization: l2  or  bending.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseRegistrationModel


# ---------------------------------------------------------------------------
# L2-normalised convolution — weights projected onto the unit sphere
# ---------------------------------------------------------------------------

class L2NormConv2d(nn.Module):
    """
    Conv2d whose filters are L2-normalised over (in_channels × kH × kW)
    before every forward pass.

    The raw weight tensor is an unconstrained nn.Parameter; normalisation
    happens only at call time so gradients flow through F.normalize and
    the weights are always kept on the unit hypersphere in filter space.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.stride  = stride
        self.padding = padding
        self.weight  = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.view(self.weight.shape[0], -1)   # (out_ch, in_ch*k*k)
        w = F.normalize(w, p=2, dim=1)                   # unit norm per filter
        w = w.view_as(self.weight)
        return F.conv2d(x, w, bias=None,
                        stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# Fixed positional encoding
# ---------------------------------------------------------------------------

def _make_pe(H: int, W: int, n_pe: int,
             device=None, dtype=None) -> Tensor:
    """
    Build a fixed sinusoidal positional encoding for a [-1, 1]^2 grid.

    n_pe must be divisible by 4.  n_pe // 4 frequencies are used, each
    contributing  [cos(f*x), sin(f*x), cos(f*y), sin(f*y)].
    Frequencies:  f_k = pi * 2^k  for k = 0 .. n_pe//4 - 1.

    Returns (1, n_pe, H, W).
    """
    if n_pe % 4 != 0:
        raise ValueError(f"n_pe must be divisible by 4, got {n_pe}")
    n_freqs = n_pe // 4

    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each

    freqs = math.pi * (2.0 ** torch.arange(n_freqs, dtype=dtype, device=device))

    channels: List[Tensor] = []
    for f in freqs.unbind():
        channels += [
            torch.cos(f * grid_x),
            torch.sin(f * grid_x),
            torch.cos(f * grid_y),
            torch.sin(f * grid_y),
        ]

    return torch.stack(channels, dim=0).unsqueeze(0)  # (1, n_pe, H, W)


# ---------------------------------------------------------------------------
# Velocity network
# ---------------------------------------------------------------------------

class GateFlowVelocityNet(nn.Module):
    """
    Gated convolutional velocity field predictor.

    Args:
        in_channels:          image channels (1 = grayscale)
        hidden_channels (C):  stem output half-width and gate channel count
        kernel_size     (k):  square stem conv kernel
        stride          (s):  stem conv stride  (controls coarse resolution)
        bottleneck_channels (K): inner width of the tanh bottleneck
        n_pe            (n):  positional-encoding channels (divisible by 4)
        image_size:           [H, W] — used to precompute the PE buffer;
                              if None the PE is computed on-the-fly each call
    """

    def __init__(
        self,
        in_channels:         int,
        hidden_channels:     int,
        kernel_size:         int,
        stride:              int,
        bottleneck_channels: int,
        n_pe:                int,
        image_size:          Optional[List[int]] = None,
    ):
        super().__init__()
        C, K, n       = hidden_channels, bottleneck_channels, n_pe
        self.C         = C
        self.n_pe      = n
        self.kernel_size = kernel_size
        self.stride    = stride
        # Zero-padding sized so the output is exactly H/s × W/s
        self.padding   = (kernel_size - stride) // 2

        # Stem — separate L2-normalised filters, no bias, zero padding
        self.conv_f = L2NormConv2d(in_channels, 2 * C, kernel_size,
                                   stride=stride, padding=self.padding)
        self.conv_g = L2NormConv2d(in_channels, 2 * C, kernel_size,
                                   stride=stride, padding=self.padding)

        # Gate bottleneck
        self.squeeze = nn.Conv2d(C, K, 1)
        self.expand  = nn.Conv2d(K, C, 1)

        # Positional-encoding branch
        self.pe_in    = nn.Conv2d(n, C, 1)   # project PE into gate space
        self.pe_out   = nn.Conv2d(C, n, 1)   # project back to PE space
        self.vel_head = nn.Conv2d(n, 2, 1)   # PE residual -> 2-channel velocity

        # Pre-compute the PE buffer if image_size is known
        if image_size is not None:
            H, W = int(image_size[0]), int(image_size[1])
            H_p = (H + 2 * self.padding - kernel_size) // stride + 1
            W_p = (W + 2 * self.padding - kernel_size) // stride + 1
            self.register_buffer("_pe_buf", _make_pe(H_p, W_p, n))
        else:
            self._pe_buf = None

    def forward(self, f: Tensor, g: Tensor) -> Tensor:
        H, W = f.shape[-2], f.shape[-1]

        # --- Stem ---
        ff = self.conv_f(f)   # (N, 2C, H', W')
        gf = self.conv_g(g)   # (N, 2C, H', W')

        # --- Gate ---
        prod = ff * gf                              # (N, 2C, H', W')
        gate = prod[:, :self.C] + prod[:, self.C:]  # (N,  C, H', W')

        # --- Bottleneck ---
        gate = self.expand(torch.tanh(self.squeeze(gate)))  # (N, C, H', W')

        # --- Positional encoding ---
        H_p, W_p = gate.shape[-2], gate.shape[-1]
        if (self._pe_buf is not None
                and self._pe_buf.shape[-2] == H_p
                and self._pe_buf.shape[-1] == W_p):
            pe = self._pe_buf.to(dtype=f.dtype)     # (1, n, H', W')  cached
        else:
            pe = _make_pe(H_p, W_p, self.n_pe, device=f.device, dtype=f.dtype)

        x = self.pe_in(pe)      # (1, C, H', W')  — broadcasts over batch
        x = x * gate            # (N, C, H', W')
        x = self.pe_out(x)      # (N, n, H', W')
        x = x - pe              # delta PE  (N, n, H', W')
        v = self.vel_head(x)    # (N, 2, H', W')

        # --- Upsample to original resolution ---
        return F.interpolate(v, size=(H, W), mode="bilinear", align_corners=True)


# ---------------------------------------------------------------------------
# Spatial warp
# ---------------------------------------------------------------------------

def _warp(f: Tensor, phi: Tensor) -> Tensor:
    """
    Warp image f by displacement field phi.

    phi : (B, 2, H, W) — displacement in normalised [-1, 1] coordinates.
          channel 0 = x (horizontal), channel 1 = y (vertical).

    A pixel at grid position (x, y) is sampled from f at (x + phi_x, y + phi_y).
    """
    B, _, H, W = f.shape
    ys = torch.linspace(-1.0, 1.0, H, device=f.device, dtype=f.dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=f.device, dtype=f.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    identity = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)
    grid = (identity + phi).permute(0, 2, 3, 1)                   # (B, H, W, 2)
    return F.grid_sample(f, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GateFlow(BaseRegistrationModel):
    """
    GateFlow: Iterative Lagrangian image registration with a gated
    convolutional displacement network and positional-encoding residual.

    At each of n_t steps the network predicts an incremental displacement
    delta from the current warped image and the fixed target.  The total
    deformation is accumulated additively — valid because each delta is
    kept small by regularisation (addition ≈ composition for small fields).

    Args:
        in_channels:          image channels (1 = grayscale)
        hidden_channels (C):  stem and gate feature width
        kernel_size     (k):  square stem conv kernel size
        stride          (s):  stem conv stride
        bottleneck_channels (K): tanh bottleneck width
        n_pe:                 positional-encoding channels (divisible by 4)
        n_t:                  number of iterative refinement steps
        image_size:           [H, W] injected by the training script
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
        image_size:          Optional[List[int]] = None,
    ):
        super().__init__()
        self.n_t = n_t

        self.velocity_net = GateFlowVelocityNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            n_pe=n_pe,
            image_size=image_size,
        )

        self._init_kwargs: Dict[str, Any] = dict(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            n_pe=n_pe,
            n_t=n_t,
            image_size=list(image_size) if image_size is not None else None,
        )

    # ------------------------------------------------------------------

    def forward(self, moving: Tensor, fixed: Tensor) -> Dict[str, Tensor]:
        B, _, H, W = moving.shape
        phi       = moving.new_zeros(B, 2, H, W)   # accumulated displacement
        f_current = moving
        trajectory = [moving]

        for _ in range(self.n_t):
            delta     = self.velocity_net(f_current, fixed) / self.n_t  # (B, 2, H, W)
            phi       = phi + delta
            f_current = _warp(moving, phi)   # always warp the original moving image
            trajectory.append(f_current)

        return {
            "warped":     f_current,
            "flow":       phi,
            "phi":        phi,
            "trajectory": torch.stack(trajectory),   # (n_t+1, B, 1, H, W)
        }

    def get_config(self) -> Dict[str, Any]:
        return dict(self._init_kwargs)

    def parameter_summary(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        vn = self.velocity_net
        return (
            f"GateFlow  C={vn.C}  K={vn.squeeze.out_channels}  "
            f"k={vn.kernel_size}  s={vn.stride}  n_pe={vn.n_pe}  "
            f"n_t={self.n_t}\n"
            f"Parameters: {total:,} total  {trainable:,} trainable"
        )
