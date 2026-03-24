"""Spatial transformer (differentiable warping) and diffeomorphic integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    Differentiable spatial transformer that warps an image by a displacement field.
    Works for both 2D (B, C, H, W) and 3D (B, C, D, H, W) inputs.
    """

    def __init__(self, size: tuple, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

        # Pre-compute base identity grid and register as buffer (not a parameter)
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)           # (ndim, *size)
        grid = grid.unsqueeze(0).float()    # (1, ndim, *size)
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src:  (B, C, *size) image to warp
            flow: (B, ndim, *size) displacement field
        Returns:
            warped image (B, C, *size)
        """
        new_locs = self.grid + flow

        # Normalize coordinates to [-1, 1] for grid_sample
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i] = 2 * (new_locs[:, i] / (shape[i] - 1) - 0.5)

        # grid_sample expects (B, *size, ndim) with xyz order
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode,
                             padding_mode="border")


class VecInt(nn.Module):
    """
    Integrates a velocity field into a diffeomorphic displacement field
    using scaling-and-squaring (Arsigny et al., 2006).
    """

    def __init__(self, size: tuple, nsteps: int = 7):
        super().__init__()
        assert nsteps >= 0
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(size)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        flow = flow * self.scale
        for _ in range(self.nsteps):
            flow = flow + self.transformer(flow, flow)
        return flow
