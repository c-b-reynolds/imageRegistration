"""
Neural ODE registration via the transport equation.

Solves the PDE:
    df/dt = v(f(r,t), g(r)) · ∇f(r,t),    f(r, 0) = moving image

where:
    f(r, t)  — image evolving from moving (t=0) toward fixed (t=1)
    g(r)     — fixed target image (constant)
    v        — velocity field predicted by a neural network
    ∇f       — spatial gradient of the current image state
    ·        — dot product (sum over spatial dimensions)

At t=1, f(r, 1) should approximate g(r).

The velocity network v takes the concatenated [f(r,t), g(r)] as input and
outputs a 2D (or 3D) vector field. This is learned end-to-end by minimising
a similarity loss between f(r, 1) and g(r).

For regularization, the velocity field at t=0 (v(f_0, g)) is returned as
'flow' so the existing RegistrationLoss can apply smoothness penalties.

Dependencies:
    pip install torchdiffeq
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint

from .base import BaseRegistrationModel


# ---------------------------------------------------------------------------
# Spatial gradient (central differences, replicate padding)
# ---------------------------------------------------------------------------

def _spatial_gradient(f: torch.Tensor) -> torch.Tensor:
    """
    Compute ∇f via central finite differences.

    Args:
        f: (B, 1, H, W)
    Returns:
        (B, 2, H, W)  — channels are (∂f/∂x, ∂f/∂y)
    """
    fp_x = F.pad(f, (1, 1, 0, 0), mode="replicate")
    fp_y = F.pad(f, (0, 0, 1, 1), mode="replicate")
    dfdx = (fp_x[:, :, :, 2:] - fp_x[:, :, :, :-2]) / 2.0
    dfdy = (fp_y[:, :, 2:, :] - fp_y[:, :, :-2, :]) / 2.0
    return torch.cat([dfdx, dfdy], dim=1)   # (B, 2, H, W)


# ---------------------------------------------------------------------------
# Velocity network  v : (f, g) -> velocity field
# ---------------------------------------------------------------------------

class VelocityNet(nn.Module):
    """
    CNN that predicts the velocity field v(f(r,t), g(r)).

    Input:  [f, g] concatenated — (B, 2, H, W)
    Output: velocity field      — (B, 2, H, W)

    Args:
        features: number of feature maps in hidden layers
        depth:    number of hidden conv layers
    """

    def __init__(self, features: int = 32, depth: int = 4):
        super().__init__()
        layers: list = []
        in_ch = 2
        for _ in range(depth):
            layers += [
                nn.Conv2d(in_ch, features, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = features
        layers.append(nn.Conv2d(features, 2, kernel_size=1))
        self.net = nn.Sequential(*layers)

        # Small-weight init — prevents large initial deformations
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([f, g], dim=1))


# ---------------------------------------------------------------------------
# ODE right-hand side:  df/dt = v(f, g) · ∇f
# ---------------------------------------------------------------------------

class TransportODEFunc(nn.Module):
    """
    Encapsulates the RHS of the transport PDE for use with torchdiffeq.

    The fixed image g must be set as an attribute before calling odeint:
        ode_func.g = fixed_image

    nfe (number of function evaluations) is tracked for diagnostics.
    """

    def __init__(self, velocity_net: VelocityNet):
        super().__init__()
        self.velocity_net = velocity_net
        self.g: torch.Tensor = None   # set per forward pass
        self.nfe: int = 0

    def forward(self, t: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: current time (scalar tensor, unused explicitly — autonomous system)
            f: (B, 1, H, W) current image state
        Returns:
            df/dt: (B, 1, H, W)
        """
        self.nfe += 1

        v      = self.velocity_net(f, self.g)       # (B, 2, H, W)
        grad_f = _spatial_gradient(f)               # (B, 2, H, W)

        # df/dt = vx * ∂f/∂x + vy * ∂f/∂y
        dfdt = (v * grad_f).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        return dfdt


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class NeuralODERegistration(BaseRegistrationModel):
    """
    Image registration by integrating the transport equation
        df/dt = v(f(r,t), g(r)) · ∇f
    from t=0 to t=1 using an adaptive ODE solver.

    Args:
        image_size: (H, W) spatial dimensions.
        in_channels: channels per image (1 for grayscale).
        features:   hidden features in VelocityNet.
        depth:      conv layers in VelocityNet.
        method:     ODE solver — 'rk4' (fixed-step, fast) or 'dopri5' (adaptive).
        n_t:        number of time points (only used by fixed-step solvers).
        rtol/atol:  tolerances for adaptive solvers (dopri5).
        adjoint:    If True, use the adjoint method for backprop (O(1) memory).
                    If False, backprop through solver steps (O(n_t) memory).
    """

    def __init__(
        self,
        image_size: tuple,
        in_channels: int = 1,
        features: int = 32,
        depth: int = 4,
        method: str = "rk4",
        n_t: int = 10,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = False,
    ):
        super().__init__()
        self.image_size  = tuple(image_size)
        self.in_channels = in_channels
        self.features    = features
        self.depth       = depth
        self.method      = method
        self.n_t         = n_t
        self.rtol        = rtol
        self.atol        = atol
        self.adjoint     = adjoint

        self.velocity_net = VelocityNet(features=features, depth=depth)
        self.ode_func     = TransportODEFunc(self.velocity_net)

        t = torch.linspace(0.0, 1.0, n_t)
        self.register_buffer("t", t)

    # ------------------------------------------------------------------

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            moving: (B, 1, H, W) moving image  — initial condition f(r, 0)
            fixed:  (B, 1, H, W) fixed image   — target g(r)
        Returns:
            warped:     f(r, 1) — transported image at t=1
            flow:       v(f_0, g) — initial velocity field (used for regularization)
            trajectory: (n_t, B, 1, H, W) full time evolution (for visualisation)
        """
        self.ode_func.g   = fixed
        self.ode_func.nfe = 0

        # Integrate:  trajectory[i] = f(r, t[i])
        # Shape: (n_t, B, 1, H, W)
        _integrate = odeint_adjoint if self.adjoint else odeint
        trajectory = _integrate(
            self.ode_func,
            moving,
            self.t,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        warped = trajectory[-1]   # f(r, 1)

        # Initial velocity — returned as 'flow' for the regularization loss
        flow = self.velocity_net(moving, fixed)   # (B, 2, H, W)

        return {"warped": warped, "flow": flow, "trajectory": trajectory}

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "image_size":  self.image_size,
            "in_channels": self.in_channels,
            "features":    self.features,
            "depth":       self.depth,
            "method":      self.method,
            "n_t":         self.n_t,
            "rtol":        self.rtol,
            "atol":        self.atol,
            "adjoint":     self.adjoint,
        }
