"""
Eulerian Hybrid ODE registration.

Uses the same HybridVelocityNet as HybridODERegistration (GatedCorrelationStem
+ Transformer + HaarDecoder) but solves only the image transport equation:

    df/dt = v(f(r,t), g(r)) · ∇f

No deformation field phi is tracked. This avoids the Lagrangian accumulation
instability present in HybridODERegistration.

Output dict:
    warped     (B, 1, H, W)       registered image at t=1
    flow       (B, 2, H, W)       instantaneous velocity at t=0
    phi        (B, 2, H, W)       alias for flow — for compatibility with
                                  RegistrationLoss and the Trainer
    trajectory (n_t, B, 1, H, W)  image states from t=0 to t=1

Recommended loss config:
    similarity:   ncc
    regularization: l2    (smoothness penalty on the velocity field)
    # do NOT use deformation_gradient — that expects an integrated phi field
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

import torch.nn.functional as F

from .base import BaseRegistrationModel
from .hybrid_ode_registration import HybridVelocityNet
from .neural_ode_registration import TransportODEFunc, _spatial_gradient


# ---------------------------------------------------------------------------
# ODE RHS with Gaussian-smoothed gradient
# ---------------------------------------------------------------------------

class SmoothedTransportODEFunc(nn.Module):
    """
    Transport ODE  df/dt = v(f, g) · ∇f_smooth
    where ∇f_smooth is the spatial gradient of a Gaussian-smoothed f.

    Smoothing suppresses speckle noise in the gradient without changing the
    image state f itself or the velocity network input.

    The fixed 3x3 Gaussian kernel is registered as a buffer (not a parameter).
    """

    def __init__(self, velocity_net: nn.Module):
        super().__init__()
        self.velocity_net = velocity_net
        self.g:   torch.Tensor = None
        self.nfe: int = 0

        kernel = torch.tensor(
            [[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]
        ) / 16.0
        # Shape (1, 1, 3, 3) — applied depthwise to a single-channel image
        self.register_buffer("gaussian", kernel.view(1, 1, 3, 3))

    def _smooth(self, f: torch.Tensor) -> torch.Tensor:
        """Apply 3x3 Gaussian to (B, 1, H, W), preserving spatial size."""
        return F.conv2d(f, self.gaussian, padding=1)

    def forward(self, t: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        v      = self.velocity_net(f, self.g)
        grad_f = _spatial_gradient(self._smooth(f))
        return (v * grad_f).sum(dim=1, keepdim=True)


class EulerianHybridODERegistration(BaseRegistrationModel):
    """
    Eulerian transport ODE with the Hybrid velocity network.

    Identical constructor to HybridODERegistration so the same YAML config
    can be used — just change model.name to 'EulerianHybridODERegistration'.

    Use regularization: l2 or bending in the loss config.
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
        # ODE
        method: str = "rk4",
        n_t: int = 10,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = False,
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
        self.method             = method
        self.n_t                = n_t
        self.rtol               = rtol
        self.atol               = atol
        self.adjoint            = adjoint

        self.velocity_net = HybridVelocityNet(
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
        self.ode_func = SmoothedTransportODEFunc(self.velocity_net)
        self.register_buffer("t", torch.linspace(0.0, 1.0, n_t))

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        self.ode_func.g   = fixed
        self.ode_func.nfe = 0

        _integrate = odeint_adjoint if self.adjoint else odeint
        trajectory = _integrate(
            self.ode_func, moving, self.t,
            method=self.method, rtol=self.rtol, atol=self.atol,
        )

        warped = trajectory[-1]
        flow   = self.velocity_net(moving, fixed)

        return {
            "warped":     warped,
            "flow":       flow,
            "phi":        flow,   # trainer/loss compatibility
            "trajectory": trajectory,
        }

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
            "method":              self.method,
            "n_t":                 self.n_t,
            "rtol":                self.rtol,
            "atol":                self.atol,
            "adjoint":             self.adjoint,
        }
