"""
Unit tests for evaluation metrics and loss functions.

Run: pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    bootstrap_ci,
    dice,
    jacobian_determinant_stats,
    mse,
    ncc,
    psnr,
)
from src.training.losses import (
    BendingEnergyLoss,
    GradientSmoothnessLoss,
    NCC,
    MSE,
    RegistrationLoss,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestNCC:
    def test_identical_images(self):
        img = np.random.rand(32, 32).astype(np.float32)
        assert abs(ncc(img, img) - 1.0) < 1e-5

    def test_negated_image(self):
        img = np.random.rand(32, 32).astype(np.float32)
        assert abs(ncc(img, -img) + 1.0) < 1e-5

    def test_range(self):
        a = np.random.rand(50, 50).astype(np.float32)
        b = np.random.rand(50, 50).astype(np.float32)
        val = ncc(a, b)
        assert -1.0 <= val <= 1.0

    def test_torch_input(self):
        a = torch.rand(1, 1, 32, 32)
        b = torch.rand(1, 1, 32, 32)
        val = ncc(a, b)
        assert isinstance(val, float)


class TestMSE:
    def test_zero_for_identical(self):
        img = np.random.rand(16, 16).astype(np.float32)
        assert mse(img, img) < 1e-10

    def test_positive(self):
        a = np.zeros((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        assert abs(mse(a, b) - 1.0) < 1e-6


class TestPSNR:
    def test_identical_returns_inf(self):
        img = np.random.rand(16, 16).astype(np.float32)
        val = psnr(img, img)
        assert val == float("inf")

    def test_higher_is_better(self):
        ref  = np.random.rand(32, 32).astype(np.float32)
        noisy_small = ref + 0.01 * np.random.randn(32, 32).astype(np.float32)
        noisy_large = ref + 0.1  * np.random.randn(32, 32).astype(np.float32)
        assert psnr(noisy_small, ref) > psnr(noisy_large, ref)


class TestDice:
    def test_perfect_overlap(self):
        seg = np.array([0, 1, 2, 1, 0, 2])
        result = dice(seg, seg)
        assert abs(result["mean_dice"] - 1.0) < 1e-6

    def test_no_overlap(self):
        pred   = np.array([1, 1, 1])
        target = np.array([2, 2, 2])
        result = dice(pred, target, labels=[1])
        assert result["dice_1"] == pytest.approx(0.0)

    def test_background_excluded(self):
        pred   = np.array([0, 1, 0])
        target = np.array([0, 1, 0])
        result = dice(pred, target)
        assert "dice_0" not in result  # background excluded

    def test_specific_labels(self):
        pred   = np.array([1, 2, 3, 1])
        target = np.array([1, 2, 3, 2])
        result = dice(pred, target, labels=[1, 2])
        assert "dice_1" in result
        assert "dice_2" in result
        assert "dice_3" not in result


class TestJacobianDet:
    def test_identity_field_2d(self):
        """Zero displacement -> Jacobian det should be ~1 everywhere."""
        flow = np.zeros((2, 32, 32), dtype=np.float32)
        stats = jacobian_determinant_stats(flow)
        assert abs(stats["jac_mean"] - 1.0) < 1e-5
        assert stats["jac_pct_neg"] == pytest.approx(0.0)

    def test_identity_field_3d(self):
        flow = np.zeros((3, 16, 16, 16), dtype=np.float32)
        stats = jacobian_determinant_stats(flow)
        assert abs(stats["jac_mean"] - 1.0) < 1e-5

    def test_output_keys(self):
        flow  = np.zeros((2, 16, 16))
        stats = jacobian_determinant_stats(flow)
        for key in ("jac_mean", "jac_std", "jac_min", "jac_max", "jac_pct_neg"):
            assert key in stats

    def test_batch_dim_stripped(self):
        """(1, ndim, *spatial) should be accepted."""
        flow  = np.zeros((1, 3, 8, 8, 8))
        stats = jacobian_determinant_stats(flow)
        assert "jac_mean" in stats


class TestBootstrapCI:
    def test_output_keys(self):
        vals = np.random.rand(50)
        out  = bootstrap_ci(vals, n=100, seed=0)
        assert set(out) == {"mean", "std", "ci_lo", "ci_hi"}

    def test_ci_contains_mean(self):
        vals = np.random.rand(100)
        out  = bootstrap_ci(vals, n=500, seed=0)
        assert out["ci_lo"] <= out["mean"] <= out["ci_hi"]

    def test_deterministic(self):
        vals = np.random.rand(30)
        a = bootstrap_ci(vals, n=100, seed=42)
        b = bootstrap_ci(vals, n=100, seed=42)
        assert a["ci_lo"] == b["ci_lo"]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestNCCLoss:
    def test_identical_images_low_loss(self):
        img  = torch.rand(1, 1, 64, 64)
        loss = NCC()(img, img)
        assert loss.item() < -0.9  # NCC -> 1, negated loss -> -1

    def test_differentiable(self):
        img1 = torch.rand(1, 1, 32, 32, requires_grad=True)
        img2 = torch.rand(1, 1, 32, 32)
        loss = NCC()(img1, img2)
        loss.backward()
        assert img1.grad is not None

    def test_3d_input(self):
        img  = torch.rand(1, 1, 16, 16, 16)
        loss = NCC(win=7)(img, img)
        assert loss.item() < -0.9


class TestGradSmoothnessLoss:
    def test_zero_for_constant_flow(self):
        flow = torch.ones(1, 2, 32, 32)
        loss = GradientSmoothnessLoss()(flow)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive(self):
        flow = torch.randn(1, 3, 16, 16, 16)
        loss = GradientSmoothnessLoss()(flow)
        assert loss.item() >= 0.0


class TestBendingEnergyLoss:
    def test_zero_for_affine_flow(self):
        """Pure translation (constant flow) should have zero bending energy."""
        flow = torch.ones(1, 2, 32, 32) * 3.0
        loss = BendingEnergyLoss()(flow)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive(self):
        flow = torch.randn(1, 3, 8, 8, 8)
        loss = BendingEnergyLoss()(flow)
        assert loss.item() >= 0.0


class TestRegistrationLoss:
    @pytest.fixture
    def loss_fn(self):
        return RegistrationLoss(similarity="ncc", regularization="bending",
                                similarity_weight=1.0, regularization_weight=0.01)

    def test_output_keys(self, loss_fn):
        w = torch.rand(1, 1, 32, 32)
        f = torch.rand(1, 1, 32, 32)
        flow = torch.randn(1, 2, 32, 32) * 0.1
        out = loss_fn(w, f, flow)
        assert {"total", "similarity", "regularization"} <= set(out)

    def test_total_is_weighted_sum(self, loss_fn):
        w = torch.rand(1, 1, 32, 32)
        f = torch.rand(1, 1, 32, 32)
        flow = torch.randn(1, 2, 32, 32) * 0.1
        out = loss_fn(w, f, flow)
        assert out["total"].requires_grad

    def test_unknown_similarity_raises(self):
        with pytest.raises(ValueError, match="Unknown similarity"):
            RegistrationLoss(similarity="unknown")

    def test_mse_similarity(self):
        fn = RegistrationLoss(similarity="mse", regularization="none")
        w  = torch.rand(1, 1, 16, 16)
        f  = w.clone()
        flow = torch.zeros(1, 2, 16, 16)
        out  = fn(w, f, flow)
        assert out["similarity"].item() == pytest.approx(0.0, abs=1e-6)
