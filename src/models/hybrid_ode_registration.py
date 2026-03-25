"""
Hybrid Neural ODE registration.

VelocityNet architecture:
    [f(r,t), g(r)]
        → GatedCorrelationStem      (L2-normalized cross-gating + pairwise pool)
        → Transformer encoder/decoder
        → HaarDecoder               (wavelet-based spatial upsampling)
        → v(r, t)                   (2D velocity field)

Integrated inside TransportODEFunc (reused from neural_ode_registration) to solve:
    df/dt = v(f(r,t), g(r)) · ∇f,    f(r,0) = moving image

Spatial resolution constraints
-------------------------------
The stem downsamples by a factor of (image_size / H'), which must be a power of 2
so that the HaarDecoder can exactly recover the original resolution.

For stride = kernel_size (non-overlapping, padding=0):
    H' = image_size // kernel_size
    upsampling factor = kernel_size   ← must be a power of 2 (4, 8, 16, 32)

For stride = kernel_size // 2 (overlapping, padding = kernel_size // 4):
    H' = 2 * image_size // kernel_size
    upsampling factor = kernel_size // 2   ← must be a power of 2

embed_dim must be divisible by 4^n_stages where n_stages = log2(upsampling factor).
"""

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint

from .base import BaseRegistrationModel
from .neural_ode_registration import _spatial_gradient


# ---------------------------------------------------------------------------
# L2-normalized convolution (no bias)
# ---------------------------------------------------------------------------

class L2NormConv2d(nn.Module):
    """
    Conv2d with no bias and per-output-channel L2-normalized weights.

    Each filter is projected onto the unit hypersphere before every forward
    pass, making the response purely direction-sensitive (cosine similarity).
    Gradients flow through the normalization at training time.

    Args:
        in_channels:  input channels
        out_channels: number of filters (large recommended)
        kernel_size:  spatial filter size
        stride:       convolution stride
        padding:      zero-padding (default 0)
        eps:          numerical stability term in normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.stride  = stride
        self.padding = padding
        self.eps     = eps
        self.weight  = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight)

    @property
    def _normalized_weight(self) -> torch.Tensor:
        w    = self.weight
        norm = w.view(w.shape[0], -1).norm(dim=1).view(w.shape[0], 1, 1, 1)
        return w / (norm + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self._normalized_weight, bias=None,
                        stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# Gated correlation stem
# ---------------------------------------------------------------------------

class GatedCorrelationStem(nn.Module):
    """
    Encodes the relationship between f and g via multiplicative gating.

    Pipeline:
        f → conv_f (L2-norm, no bias) ─┐
                                        ├─ ⊙ → gate → pairwise_pool
        g → conv_g (L2-norm, no bias) ─┘          → Conv1x1 → sigmoid
                                                   → token_conv → tokens

    The element-wise product (gate) fires where both images respond
    strongly in the same filter direction — analogous to spectral
    cross-correlation normalised by magnitude.

    Pairwise channel pooling compresses consecutive channel pairs [0,1],
    [2,3], ... down to C/2 channels using max or average.

    Args:
        in_channels:   channels per image (1 for grayscale)
        stem_channels: filters in conv_f / conv_g (must be even, recommend large)
        kernel_size:   spatial extent of each filter
        stride:        convolution stride
        padding:       convolution padding
        pool_type:     'max' or 'avg' for pairwise channel compression
        hidden_dim:    channels after 1×1 conv + sigmoid
        embed_dim:     output token channels fed to transformer
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_type: str,
        hidden_dim: int,
        embed_dim: int,
    ):
        super().__init__()
        assert stem_channels % 2 == 0, "stem_channels must be even for pairwise pooling"
        assert pool_type in ("max", "avg"), "pool_type must be 'max' or 'avg'"

        self.pool_type = pool_type
        self.conv_f    = L2NormConv2d(in_channels, stem_channels, kernel_size, stride, padding)
        self.conv_g    = L2NormConv2d(in_channels, stem_channels, kernel_size, stride, padding)
        self.proj      = nn.Conv2d(stem_channels // 2, hidden_dim, kernel_size=1)
        self.token_conv = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)

    def _pairwise_pool(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, C/2, H, W) via pairwise channel pool."""
        B, C, H, W = x.shape
        x = x.view(B, C // 2, 2, H, W)
        return x.max(dim=2).values if self.pool_type == "max" else x.mean(dim=2)

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f_feat = self.conv_f(f)                    # (B, stem_channels, H', W')
        g_feat = self.conv_g(g)                    # (B, stem_channels, H', W')
        gate   = f_feat * g_feat                   # (B, stem_channels, H', W')
        pooled = self._pairwise_pool(gate)         # (B, stem_channels//2, H', W')
        hidden = torch.sigmoid(self.proj(pooled))  # (B, hidden_dim, H', W')
        return self.token_conv(hidden)             # (B, embed_dim, H', W')


# ---------------------------------------------------------------------------
# Haar wavelet decoder
# ---------------------------------------------------------------------------

class HaarWaveletUnsqueeze(nn.Module):
    """
    Fixed inverse Haar wavelet transform.
    Input:  (B, 4C, H, W)
    Output: (B, C,  2H, 2W)
    Channels are quartered; spatial resolution is doubled.
    """

    def __init__(self):
        super().__init__()
        haar = torch.tensor([
            [[[0.5,  0.5], [0.5,  0.5]]],   # LL
            [[[0.5, -0.5], [0.5, -0.5]]],   # LH
            [[[0.5,  0.5], [-0.5, -0.5]]],  # HL
            [[[0.5, -0.5], [-0.5, 0.5]]],   # HH
        ])
        self.register_buffer("kernel", haar)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        B, C4, H, W = y.shape
        assert C4 % 4 == 0, f"Input channels must be divisible by 4, got {C4}"
        C      = C4 // 4
        weight = self.kernel.repeat(C, 1, 1, 1)
        return F.conv_transpose2d(y, weight=weight, bias=None,
                                  stride=2, padding=0, groups=C)


class HaarDecoder(nn.Module):
    """
    Upsamples transformer output to full image resolution.

    The number of stages is computed at init from the stem's downsampling
    factor: n_stages = log2(image_size / H_prime).

    Per stage:
        Conv2d(ch, ch, 3, pad=1) → HaarWaveletUnsqueeze → [GELU if not last]
    Final:
        Conv2d(ch_final, out_channels, 1)

    Constraint: embed_dim must be divisible by 4^n_stages.
    """

    def __init__(self, embed_dim: int, out_channels: int, n_stages: int):
        super().__init__()
        divisor = 4 ** n_stages
        if embed_dim % divisor != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by 4^{n_stages}={divisor}. "
                f"Minimum valid embed_dim for {n_stages} stages: {divisor}."
            )
        layers = []
        ch = embed_dim
        for i in range(n_stages):
            layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1))
            layers.append(HaarWaveletUnsqueeze())
            ch //= 4
            if i < n_stages - 1:
                layers.append(nn.GELU())
        out_conv = nn.Conv2d(ch, out_channels, kernel_size=1)
        nn.init.zeros_(out_conv.weight)
        nn.init.zeros_(out_conv.bias)
        layers.append(out_conv)
        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


# ---------------------------------------------------------------------------
# Hybrid velocity network
# ---------------------------------------------------------------------------

def _compute_h_prime(image_size: int, kernel_size: int, stride: int, padding: int) -> int:
    return math.floor((image_size + 2 * padding - kernel_size) / stride) + 1


class HybridVelocityNet(nn.Module):
    """
    Full velocity network: GatedCorrelationStem → Transformer → HaarDecoder.
    """

    def __init__(
        self,
        image_size: tuple,
        in_channels: int,
        stem_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_type: str,
        stem_hidden_dim: int,
        embed_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        nhead: int,
        ffn_dim: int,
        tgt_mode: str,
        dropout: float,
    ):
        super().__init__()
        self.tgt_mode = tgt_mode

        # Spatial resolution after stem
        H_prime = _compute_h_prime(image_size[0], kernel_size, stride, padding)
        W_prime = _compute_h_prime(image_size[1], kernel_size, stride, padding)
        seq_len = H_prime * W_prime

        # Haar stages from downsampling ratio
        ratio = image_size[0] / H_prime
        n_stages = int(round(math.log2(ratio)))
        if abs(2 ** n_stages - ratio) > 1e-6:
            raise ValueError(
                f"Upsampling ratio {ratio:.4f} is not a power of 2. "
                f"Use stride=kernel_size (non-overlapping) or "
                f"stride=kernel_size//2 with padding=kernel_size//4 (overlapping)."
            )

        # ---- Stem ----
        self.stem = GatedCorrelationStem(
            in_channels=in_channels,
            stem_channels=stem_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool_type=pool_type,
            hidden_dim=stem_hidden_dim,
            embed_dim=embed_dim,
        )

        # ---- Positional embedding ----
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ---- Learned target queries (tgt_mode='learned') ----
        self.tgt_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.tgt_embed, std=0.02)

        # ---- Transformer ----
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=nhead, dim_feedforward=ffn_dim,
                dropout=dropout, batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=nhead, dim_feedforward=ffn_dim,
                dropout=dropout, batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )

        # ---- Haar decoder ----
        self.haar_decoder = HaarDecoder(
            embed_dim=embed_dim,
            out_channels=2,
            n_stages=n_stages,
        )

        self._H_prime = H_prime
        self._W_prime = W_prime

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B = f.shape[0]

        # Stem → tokens: (B, embed_dim, H', W')
        tokens = self.stem(f, g)

        # Flatten + positional embedding: (B, H'W', embed_dim)
        tokens = tokens.flatten(2).permute(0, 2, 1) + self.pos_embed

        # Transformer
        memory  = self.encoder(tokens)
        tgt     = memory if self.tgt_mode == "encoder_output" \
                  else self.tgt_embed.expand(B, -1, -1)
        decoded = self.decoder(tgt, memory)             # (B, H'W', embed_dim)

        # Reshape to spatial: (B, embed_dim, H', W')
        decoded = decoded.permute(0, 2, 1).view(B, -1, self._H_prime, self._W_prime)

        # Haar decode → velocity field: (B, 2, H, W)
        return self.haar_decoder(decoded)


# ---------------------------------------------------------------------------
# Identity grid and deformation field utilities
# ---------------------------------------------------------------------------

def _make_identity_grid(
    B: int, H: int, W: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Return the identity deformation field in normalized [-1, 1] coordinates.

    Output shape: (B, 2, H, W)
      channel 0: x-coordinates (horizontal, left=-1, right=+1)
      channel 1: y-coordinates (vertical,   top=-1,  bottom=+1)
    Matches the convention expected by F.grid_sample with align_corners=True.
    """
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")          # (H, W) each
    identity = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)    # (1, 2, H, W)
    return identity.expand(B, -1, -1, -1)


def deformation_gradient_loss(phi: torch.Tensor) -> torch.Tensor:
    """
    Regularization loss on the final deformation field.

    Computes the mean squared Frobenius norm of the Jacobian of the
    displacement field u = phi - identity, where displacements are expressed
    in pixel units. This gives a dimensionless strain (pixels/pixel) that is
    naturally on the same scale as NCC loss for typical image deformations.

    Specifically:
        loss = mean( (∂u_x/∂x)² + (∂u_x/∂y)² + (∂u_y/∂x)² + (∂u_y/∂y)² )

    where gradients are forward differences in pixel-index space and u is
    in pixels (converted from normalized [-1,1] coordinates).

    To include in total loss:
        loss_total = loss_ncc + reg_factor * deformation_gradient_loss(phi)

    Parameters
    ----------
    phi : torch.Tensor  (B, 2, H, W)
        Final deformation field in normalized [-1, 1] coordinates, as returned
        in the 'phi' key of HybridODERegistration.forward().

    Returns
    -------
    scalar tensor
    """
    B, _, H, W = phi.shape

    identity = _make_identity_grid(B, H, W, phi.device, phi.dtype)
    u_norm = phi - identity  # displacement in normalized [-1, 1] coords

    # Convert to pixel displacement so the gradient is dimensionless strain
    scale = phi.new_tensor([(W - 1) / 2.0, (H - 1) / 2.0]).view(1, 2, 1, 1)
    u = u_norm * scale  # (B, 2, H, W) in pixels

    # Forward differences: change in pixel displacement per one-pixel step
    du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]   # (B, 2, H, W-1)
    du_dy = u[:, :, 1:, :] - u[:, :, :-1, :]   # (B, 2, H-1, W)

    return du_dx.pow(2).mean() + du_dy.pow(2).mean()


# ---------------------------------------------------------------------------
# Augmented ODE function — integrates image and deformation field jointly
# ---------------------------------------------------------------------------

class AugmentedTransportODEFunc(nn.Module):
    """
    Extends the transport ODE to also integrate the deformation field φ.

    State: (f, phi)
      f   : (B, 1, H, W)  moving image being advected
      phi : (B, 2, H, W)  deformation field in normalized [-1, 1] coords

    ODEs:
      df/dt   = v(f, g) · ∇f          (image transport)
      dφ/dt   = v(t, φ(r, t))         (Lagrangian tracking)

    The Lagrangian update samples the current velocity field at the deformed
    positions φ via bilinear interpolation, giving the true continuous flow
    map rather than a sum-of-velocities approximation.

    Initialize phi as the identity grid before calling odeint.
    Set ode_func.g = fixed_image before each forward pass.
    """

    def __init__(self, velocity_net: nn.Module):
        super().__init__()
        self.velocity_net = velocity_net
        self.g:   torch.Tensor = None
        self.nfe: int = 0

    def forward(
        self, t: torch.Tensor, state: tuple
    ) -> tuple:
        f, phi = state
        self.nfe += 1

        v = self.velocity_net(f, self.g)             # (B, 2, H, W)

        # --- image transport ---
        grad_f = _spatial_gradient(f)                # (B, 2, H, W)
        dfdt   = (v * grad_f).sum(dim=1, keepdim=True)

        # --- deformation field: sample v at current deformed positions ---
        # F.grid_sample expects grid as (B, H, W, 2)
        phi_grid  = phi.permute(0, 2, 3, 1)
        v_at_phi  = F.grid_sample(
            v, phi_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )                                            # (B, 2, H, W)

        return dfdt, v_at_phi


# ---------------------------------------------------------------------------
# Full registration model
# ---------------------------------------------------------------------------

class HybridODERegistration(BaseRegistrationModel):
    """
    Neural ODE registration using the Hybrid velocity network.

    Solves df/dt = v(f(r,t), g(r)) · ∇f from t=0 to t=1.

    Stem stride modes
    -----------------
    Non-overlapping (default):  stride = kernel_size,      padding = 0
    Overlapping:                stride = kernel_size // 2, padding = kernel_size // 4
    Set these explicitly in the config — both must yield a power-of-2
    downsampling ratio relative to image_size.

    embed_dim constraint
    --------------------
    embed_dim must be divisible by 4^n_stages where
    n_stages = log2(image_size / H_prime).
    A ValueError is raised at init if this is violated.
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
        self.ode_func = AugmentedTransportODEFunc(self.velocity_net)
        self.register_buffer("t", torch.linspace(0.0, 1.0, n_t))

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = moving.shape
        self.ode_func.g   = fixed
        self.ode_func.nfe = 0

        phi0 = _make_identity_grid(B, H, W, moving.device, moving.dtype)

        _integrate = odeint_adjoint if self.adjoint else odeint
        trajectory, trajectory_phi = _integrate(
            self.ode_func, (moving, phi0), self.t,
            method=self.method, rtol=self.rtol, atol=self.atol,
        )

        warped    = trajectory[-1]       # (B, 1, H, W)
        phi_final = trajectory_phi[-1]   # (B, 2, H, W) final deformation field
        flow      = self.velocity_net(moving, fixed)
        return {"warped": warped, "flow": flow, "trajectory": trajectory, "phi": phi_final}

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
