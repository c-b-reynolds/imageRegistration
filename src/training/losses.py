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
    _REG = {"l2": GradientSmoothnessLoss, "bending": BendingEnergyLoss}

    def __init__(
        self,
        similarity: str = "ncc",
        regularization: str = "bending",
        similarity_weight: float = 1.0,
        regularization_weight: float = 0.01,
    ):
        super().__init__()
        if similarity not in self._SIM:
            raise ValueError(f"Unknown similarity '{similarity}'. Choose from {list(self._SIM)}")

        self.sim_fn = self._SIM[similarity]()
        self.sim_w = similarity_weight

        self.reg_fn = None
        self.reg_w = regularization_weight
        if regularization != "none":
            if regularization not in self._REG:
                raise ValueError(f"Unknown regularization '{regularization}'")
            self.reg_fn = self._REG[regularization]()

    def forward(
        self, warped: torch.Tensor, fixed: torch.Tensor, flow: torch.Tensor
    ) -> dict:
        sim  = self.sim_fn(warped, fixed)
        reg  = self.reg_fn(flow) if self.reg_fn is not None else torch.zeros(1, device=flow.device)
        total = self.sim_w * sim + self.reg_w * reg
        return {"total": total, "similarity": sim.detach(), "regularization": reg.detach()}
