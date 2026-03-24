"""
Unit tests for model architectures.

Run: pytest tests/test_models.py -v
"""

import pytest
import torch

from src.models import UNetRegistration, build_model
from src.models.spatial_transformer import SpatialTransformer, VecInt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def image_size_2d():
    return (64, 64)


@pytest.fixture
def image_size_3d():
    return (32, 32, 32)


@pytest.fixture
def batch_size():
    return 2


# ---------------------------------------------------------------------------
# SpatialTransformer
# ---------------------------------------------------------------------------

class TestSpatialTransformer:
    def test_identity_flow_2d(self, image_size_2d):
        """Zero flow should return the original image unchanged."""
        B, C = 1, 1
        H, W = image_size_2d
        st  = SpatialTransformer(image_size_2d)
        src = torch.rand(B, C, H, W)
        flow = torch.zeros(B, 2, H, W)
        out = st(src, flow)
        assert out.shape == src.shape
        assert torch.allclose(out, src, atol=1e-5), "Identity warp should preserve image"

    def test_identity_flow_3d(self, image_size_3d):
        B, C = 1, 1
        D, H, W = image_size_3d
        st   = SpatialTransformer(image_size_3d)
        src  = torch.rand(B, C, D, H, W)
        flow = torch.zeros(B, 3, D, H, W)
        out  = st(src, flow)
        assert out.shape == src.shape
        assert torch.allclose(out, src, atol=1e-5)

    def test_output_range(self, image_size_2d):
        B, C, H, W = 1, 1, *image_size_2d
        st   = SpatialTransformer(image_size_2d)
        src  = torch.rand(B, C, H, W)
        flow = torch.randn(B, 2, H, W) * 5.0
        out  = st(src, flow)
        # With border padding the output should stay within input range
        assert out.min() >= src.min() - 1e-5
        assert out.max() <= src.max() + 1e-5


class TestVecInt:
    def test_output_shape(self, image_size_3d):
        B = 1
        D, H, W = image_size_3d
        vi   = VecInt(image_size_3d, nsteps=7)
        flow = torch.randn(B, 3, D, H, W) * 0.1
        out  = vi(flow)
        assert out.shape == flow.shape

    def test_zero_velocity_gives_zero_displacement(self, image_size_2d):
        vi   = VecInt(image_size_2d, nsteps=7)
        flow = torch.zeros(1, 2, *image_size_2d)
        out  = vi(flow)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


# ---------------------------------------------------------------------------
# UNetRegistration
# ---------------------------------------------------------------------------

class TestUNetRegistration:
    def test_output_keys_2d(self, image_size_2d, batch_size):
        model = UNetRegistration(image_size_2d, in_channels=1, base_features=8, depth=3,
                                 int_steps=0, ndim=2)
        B, C = batch_size, 1
        H, W = image_size_2d
        moving = torch.rand(B, C, H, W)
        fixed  = torch.rand(B, C, H, W)
        out = model(moving, fixed)
        assert "warped" in out
        assert "flow" in out

    def test_output_shapes_2d(self, image_size_2d, batch_size):
        model = UNetRegistration(image_size_2d, in_channels=1, base_features=8, depth=3,
                                 int_steps=0, ndim=2)
        B, C, H, W = batch_size, 1, *image_size_2d
        moving = torch.rand(B, C, H, W)
        fixed  = torch.rand(B, C, H, W)
        out = model(moving, fixed)
        assert out["warped"].shape == (B, C, H, W)
        assert out["flow"].shape   == (B, 2, H, W)

    def test_output_shapes_3d(self, image_size_3d, batch_size):
        model = UNetRegistration(image_size_3d, in_channels=1, base_features=8, depth=3,
                                 int_steps=7, ndim=3)
        B, C, D, H, W = batch_size, 1, *image_size_3d
        moving = torch.rand(B, C, D, H, W)
        fixed  = torch.rand(B, C, D, H, W)
        out = model(moving, fixed)
        assert out["warped"].shape == (B, C, D, H, W)
        assert out["flow"].shape   == (B, 3, D, H, W)

    def test_diffeomorphic_disabled(self, image_size_2d):
        model = UNetRegistration(image_size_2d, int_steps=0, ndim=2, base_features=8, depth=3)
        assert model.vec_int is None

    def test_diffeomorphic_enabled(self, image_size_2d):
        model = UNetRegistration(image_size_2d, int_steps=7, ndim=2, base_features=8, depth=3)
        assert model.vec_int is not None

    def test_parameter_count(self, image_size_2d):
        model = UNetRegistration(image_size_2d, base_features=8, depth=3, ndim=2, int_steps=0)
        assert model.num_parameters > 0

    def test_get_config_roundtrip(self, image_size_2d):
        model = UNetRegistration(image_size_2d, in_channels=1, base_features=8, depth=3,
                                 int_steps=7, ndim=2)
        cfg     = model.get_config()
        rebuilt = UNetRegistration(**cfg)
        assert rebuilt.num_parameters == model.num_parameters

    def test_gradient_flow(self, image_size_2d):
        model  = UNetRegistration(image_size_2d, base_features=8, depth=3, ndim=2, int_steps=0)
        moving = torch.rand(1, 1, *image_size_2d, requires_grad=False)
        fixed  = torch.rand(1, 1, *image_size_2d, requires_grad=False)
        out    = model(moving, fixed)
        loss   = out["warped"].mean()
        loss.backward()
        # Check that at least one parameter received a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients reached model parameters"

    def test_build_model_registry(self, image_size_2d):
        model = build_model("UNetRegistration", image_size=image_size_2d,
                            base_features=8, depth=3, ndim=2, int_steps=0)
        assert isinstance(model, UNetRegistration)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("NonExistentNet", image_size=(32, 32))
