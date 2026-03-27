"""
Loss functions for image registration.

All losses accept batched torch tensors and return a scalar tensor.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Similarity losses
# ---------------------------------------------------------------------------

class NCC(torch.nn.Module):
    """
    Local Normalized Cross-Correlation.

    Computed over sliding windows — more robust than global NCC for
    images with regional intensity variations.

    Reference: Avants et al., "Symmetric diffeomorphic image registration
    with cross-correlation", Med Image Anal 2008.
    """

    def __init__(self, win: int = 9, eps: float = 1e-5):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ndim = y_pred.dim() - 2
        assert ndim in (2, 3), "NCC expects 2D or 3D input"

        win_size = self.win ** ndim
        sum_fn = F.avg_pool3d if ndim == 3 else F.avg_pool2d

        pad = self.win // 2
        kwargs = dict(kernel_size=self.win, stride=1, padding=pad)

        # Local means
        u_I = sum_fn(y_pred, **kwargs)
        u_J = sum_fn(y_true, **kwargs)

        # Local variances / cross-correlation
        I2  = sum_fn(y_pred * y_pred, **kwargs) - u_I * u_I
        J2  = sum_fn(y_true * y_true, **kwargs) - u_J * u_J
        IJ  = sum_fn(y_pred * y_true, **kwargs) - u_I * u_J

        ncc = (IJ * IJ) / (I2 * J2 + self.eps)
        return -ncc.mean()  # negate: we minimize


class MSE(torch.nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_pred, y_true)


class SSIM(torch.nn.Module):
    """
    Structural Similarity Index as a loss (returns 1 - SSIM).
    Computed globally per image pair; for local SSIM use a sliding window.
    """

    def __init__(self, data_range: float = 1.0, win: int = 11, eps: float = 1e-8):
        super().__init__()
        self.data_range = data_range
        self.win = win
        self.eps = eps
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        self.register_buffer = None  # not an nn.Module with buffers
        self._C1 = C1
        self._C2 = C2

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ndim = y_pred.dim() - 2
        pool_fn = F.avg_pool3d if ndim == 3 else F.avg_pool2d
        pad = self.win // 2
        kwargs = dict(kernel_size=self.win, stride=1, padding=pad)

        mu1 = pool_fn(y_pred, **kwargs)
        mu2 = pool_fn(y_true, **kwargs)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12   = mu1 * mu2

        sigma1_sq = pool_fn(y_pred ** 2, **kwargs) - mu1_sq
        sigma2_sq = pool_fn(y_true ** 2, **kwargs) - mu2_sq
        sigma12   = pool_fn(y_pred * y_true, **kwargs) - mu12

        C1, C2 = self._C1, self._C2
        numerator   = (2 * mu12 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map    = numerator / (denominator + self.eps)
        return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Regularization losses (on displacement / velocity field)
# ---------------------------------------------------------------------------

class GradientSmoothnessLoss(torch.nn.Module):
    """
    L2 penalty on spatial gradients of the displacement field.
    Encourages smooth, physically plausible deformations.

    penalty: 'l2' or 'l1'
    """

    def __init__(self, penalty: str = "l2"):
        super().__init__()
        assert penalty in ("l1", "l2")
        self.penalty = penalty

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        ndim = flow.dim() - 2
        diffs = []
        for i in range(ndim):
            # finite differences along each spatial axis
            d = i + 2  # spatial dim index
            diff = flow.narrow(d, 1, flow.shape[d] - 1) - flow.narrow(d, 0, flow.shape[d] - 1)
            diffs.append(diff)

        if self.penalty == "l2":
            return sum((d ** 2).mean() for d in diffs)
        else:
            return sum(d.abs().mean() for d in diffs)


class BendingEnergyLoss(torch.nn.Module):
    """
    Bending energy (second-order spatial regularization).
    Penalizes non-affine local deformations more aggressively than L2 gradient.
    """

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        ndim = flow.dim() - 2
        loss = torch.tensor(0.0, device=flow.device)
        for i in range(ndim):
            for j in range(ndim):
                di = i + 2
                dj = j + 2
                # Second-order finite differences (Laplacian-like)
                f_i = torch.diff(flow, n=1, dim=di)
                # Trim to allow diff along second axis
                trim = min(f_i.shape[dj], flow.shape[dj] - 1)
                f_i = f_i.narrow(dj, 0, trim)
                f_ij = torch.diff(f_i, n=1, dim=dj)
                loss = loss + (f_ij ** 2).mean()
        return loss


def _jacobian_det(phi: torch.Tensor) -> torch.Tensor:
    """
    Jacobian determinant of the full deformation F(x) = x + phi(x).

    phi : (B, 2, H, W) displacement in normalised [-1, 1] coords.

    Returns det(J_F) = (1 + d_phi_x/dx)(1 + d_phi_y/dy)
                     - (d_phi_x/dy)(d_phi_y/dx)   shape (B, 1, H, W).

    Spatial derivatives are computed via central differences with
    replicate padding so the output has the same spatial size as phi.
    """
    px = F.pad(phi, (1, 1, 0, 0), mode="replicate")
    py = F.pad(phi, (0, 0, 1, 1), mode="replicate")

    dphi_x_dx = (px[:, 0:1, :, 2:] - px[:, 0:1, :, :-2]) / 2.0
    dphi_y_dx = (px[:, 1:2, :, 2:] - px[:, 1:2, :, :-2]) / 2.0
    dphi_x_dy = (py[:, 0:1, 2:, :] - py[:, 0:1, :-2, :]) / 2.0
    dphi_y_dy = (py[:, 1:2, 2:, :] - py[:, 1:2, :-2, :]) / 2.0

    return ((1.0 + dphi_x_dx) * (1.0 + dphi_y_dy)
            - dphi_x_dy * dphi_y_dx)


class NegativeJacobianLoss(torch.nn.Module):
    """
    Penalises local folding in the deformation field.

        L = mean( relu( -(det(J_F) + eps) )^2 )

    det(J_F) < 0 means the mapping has folded at that location.
    The relu^2 is zero where the mapping is well-behaved, concentrating
    learning on fold removal only.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        det = _jacobian_det(phi)
        return F.relu(-(det + self.eps)).pow(2).mean()


class LogJacobianLoss(torch.nn.Module):
    """
    Log-determinant Jacobian regularization.

        L = mean( (log det(J_F))^2 )

    Penalises any deviation from det = 1 (log = 0), encouraging a
    volume-preserving (incompressible) transformation.  Used in LDDMM
    and ANTs for tissues such as brain where volume is approximately
    conserved.

    det is clamped to clamp_min before taking the log to avoid
    instability at or below zero (folded regions).
    """

    def __init__(self, clamp_min: float = 1e-5):
        super().__init__()
        self.clamp_min = clamp_min

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        det = _jacobian_det(phi)
        return torch.log(det.clamp(min=self.clamp_min)).pow(2).mean()


class DeformationGradientLoss(torch.nn.Module):
    """
    Regularization on the integrated deformation field phi (B, 2, H, W).

    Converts the displacement (phi - identity) to pixel units then computes
    the mean squared Frobenius norm of its Jacobian via forward differences.
    The result is a dimensionless strain (pixels/pixel) on the same scale
    as NCC loss for typical image deformations, so reg_weight=1 is a
    meaningful starting point relative to the similarity term.
    """

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        B, _, H, W = phi.shape

        # Identity grid in normalized [-1, 1] coords
        xs = torch.linspace(-1, 1, W, device=phi.device, dtype=phi.dtype)
        ys = torch.linspace(-1, 1, H, device=phi.device, dtype=phi.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        identity = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)

        # Displacement in normalized coords → convert to pixel units
        u_norm = phi - identity
        scale  = phi.new_tensor([(W - 1) / 2.0, (H - 1) / 2.0]).view(1, 2, 1, 1)
        u      = u_norm * scale  # (B, 2, H, W) pixels

        # Forward differences: pixel displacement change per one-pixel step
        du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]   # (B, 2, H, W-1)
        du_dy = u[:, :, 1:, :] - u[:, :, :-1, :]   # (B, 2, H-1, W)

        return du_dx.pow(2).mean() + du_dy.pow(2).mean()


# ---------------------------------------------------------------------------
# Combined registration loss
# ---------------------------------------------------------------------------

class RegistrationLoss(torch.nn.Module):
    """
    Weighted combination of similarity + regularization losses.

    Args:
        similarity:           'ncc' | 'mse' | 'ssim'
        regularization:       'l2' | 'bending' | 'none'
        similarity_weight:    weight for similarity term
        regularization_weight: weight for regularization term
    """

    _SIM = {"ncc": NCC, "mse": MSE, "ssim": SSIM}
    _REG = {"l2": GradientSmoothnessLoss, "bending": BendingEnergyLoss,
            "deformation_gradient": DeformationGradientLoss}

    def __init__(
        self,
        similarity: str = "ncc",
        regularization: str = "bending",
        similarity_weight: float = 1.0,
        regularization_weight: float = 0.01,
        ncc_win: int = 9,
        jacobian_det_weight: float = 0.0,
        jacobian_eps:        float = 1e-3,
        log_jacobian_weight: float = 0.0,
        log_jacobian_clamp:  float = 1e-5,
    ):
        super().__init__()
        if similarity not in self._SIM:
            raise ValueError(f"Unknown similarity '{similarity}'. Choose from {list(self._SIM)}")

        self.sim_fn = NCC(win=ncc_win) if similarity == "ncc" else self._SIM[similarity]()
        self.sim_w  = similarity_weight

        self.reg_fn = None
        self.reg_w  = regularization_weight
        if regularization != "none":
            if regularization not in self._REG:
                raise ValueError(f"Unknown regularization '{regularization}'")
            self.reg_fn = self._REG[regularization]()

        self.jac_fn  = NegativeJacobianLoss(eps=float(jacobian_eps))   if jacobian_det_weight > 0.0 else None
        self.jac_w   = jacobian_det_weight

        self.logj_fn = LogJacobianLoss(clamp_min=float(log_jacobian_clamp)) if log_jacobian_weight > 0.0 else None
        self.logj_w  = log_jacobian_weight

    def forward(
        self, warped: torch.Tensor, fixed: torch.Tensor, phi: torch.Tensor
    ) -> dict:
        sim  = self.sim_fn(warped, fixed)
        reg  = self.reg_fn(phi)  if self.reg_fn  is not None else phi.new_zeros(1)
        jac  = self.jac_fn(phi)  if self.jac_fn  is not None else phi.new_zeros(1)
        logj = self.logj_fn(phi) if self.logj_fn is not None else phi.new_zeros(1)

        total = self.sim_w * sim + self.reg_w * reg + self.jac_w * jac + self.logj_w * logj
        return {
            "total":          total,
            "similarity":     sim.detach(),
            "regularization": reg.detach(),
            "jacobian":       jac.detach(),
            "log_jacobian":   logj.detach(),
        }
