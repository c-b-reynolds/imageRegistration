# Image Registration

A PyTorch framework for 2D image registration using neural ODEs. Designed primarily for SAR (Synthetic Aperture Radar) image pairs produced by the [sar_align](../sar_align) repo, but supports arbitrary 2D image pairs.

---

## How it works

### Registration as a transport ODE

Rather than predicting a static displacement field, the model solves a continuous transport equation from `t=0` to `t=1`:

```
df/dt = v(f(r,t), g(r)) · ∇f
```

where `f` is the moving image being advected and `g` is the fixed image. A velocity network `v` is evaluated at each ODE time step, taking the current image state and the fixed image as input. The ODE solver accumulates these steps to produce the final warped image `f(r,1)`.

### Deformation field integration

The `HybridODERegistration` model extends the ODE state to `(f, φ)`, integrating the deformation field alongside the image:

```
df/dt  = v(f, g) · ∇f           (image transport)
dφ/dt  = v(t, φ(r,t))           (Lagrangian tracking)
```

The second equation samples the velocity at the current deformed positions via bilinear interpolation, giving the true continuous flow map rather than a sum-of-velocities approximation. The final `φ` maps reference frame coordinates to their positions in the moving image at `t=1`.

### HybridODERegistration architecture

```
[f(r,t), g(r)]
    → GatedCorrelationStem     L2-normalised cross-gating + pairwise pool
    → Transformer encoder/decoder
    → HaarDecoder              wavelet-based spatial upsampling
    → v(r, t)                  2D velocity field (B, 2, H, W)
```

The **GatedCorrelationStem** computes an element-wise product of L2-normalised filter responses from each image — analogous to normalised cross-correlation — then compresses channels via pairwise pooling. This makes the stem inherently sensitive to image correspondence rather than raw intensity.

The **HaarDecoder** upsamples transformer output back to full resolution using fixed inverse Haar wavelet transforms, which exactly recover spatial resolution with no learned upsampling artifacts.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick start

### SAR patch pairs (primary use case)

```bash
# Generates dummy patches.npy automatically if absent, then trains
python scripts/train_hybrid_patches.py

# With real SAR patches from sar_align
python scripts/train_hybrid_patches.py --patches path/to/patches.npy

# Tune key hyperparameters from the command line
python scripts/train_hybrid_patches.py \
    --patches path/to/patches.npy \
    --reg-weight 0.05 \
    --epochs 50 \
    --batch-size 16 \
    --no-erasing
```

### Numpy atlas data (general images)

```bash
python scripts/generate_dummy_data.py   # creates data/ with Gaussian blob images
python scripts/train_numpy.py --config configs/experiments/hybrid_ode_numpy.yaml
```

---

## Loading your own data

### SAR patch pairs from sar_align

The `sar_align` repo produces a `patches.npy` file of shape `(N, 2, H, W)` where channel 0 is image A and channel 1 is image B, both normalised to `[0, 1]`. Pass this directly:

```python
from src.data import build_patch_dataloaders

cfg = {
    "data": {
        "patches_path": "path/to/patches.npy",
        "val_frac": 0.15,
        "test_frac": 0.15,
    },
    "training": {"batch_size": 16},
    "augmentation": {
        "use_rotation": True,
        "use_erasing":  True,
        "erasing_scale": [0.02, 0.2],
        "erasing_ratio": [0.3, 3.0],
    },
}
train_loader, val_loader, test_loader = build_patch_dataloaders(cfg)
```

The file is memory-mapped (`mmap_mode='r'`) so large datasets are never fully loaded into RAM.

### Individual numpy files

Place `(H, W)` float32 `.npy` files in `data/train/`, `data/val/`, `data/test/` and create an atlas:

```python
import numpy as np
from pathlib import Path
imgs = [np.load(p) for p in sorted(Path("data/train").glob("*.npy"))]
np.save("data/atlas.npy", np.mean(imgs, axis=0).astype(np.float32))
```

Then train with `scripts/train_numpy.py`.

### Custom format

Subclass `BaseRegistrationDataset` in `src/data/` following the pattern in `numpy_dataset.py`. Return a dict with at minimum `"moving"` and `"fixed"` tensors of shape `(1, H, W)`.

---

## Models

| Model | Description |
|-------|-------------|
| `SimpleCNN` | Minimal CNN baseline |
| `NeuralODERegistration` | Transport ODE with CNN velocity network |
| `HybridODERegistration` | Transport ODE with Gated Stem + Transformer; also tracks full deformation field |
| `UNetRegistration` | Full UNet encoder-decoder, 2D/3D capable |

All models are registered by name and can be selected via config. `HybridODERegistration` is recommended for SAR image pairs.

### HybridODERegistration output dict

```python
out = model(moving, fixed)
out["warped"]      # (B, 1, H, W)        registered image at t=1
out["phi"]         # (B, 2, H, W)        final deformation field in [-1, 1] coords
out["flow"]        # (B, 2, H, W)        instantaneous velocity at t=0 (diagnostic)
out["trajectory"]  # (n_t, B, 1, H, W)  image states from t=0 to t=1
```

### Spatial resolution constraints (HybridODERegistration)

The stem downsampling ratio `image_size / H'` must be a power of 2, and `embed_dim` must be divisible by `4^n_stages`:

| Image size | kernel_size | stride | padding | H' | n_stages | Min embed_dim |
|------------|-------------|--------|---------|----|----------|---------------|
| 64×64 | 8 | 8 | 0 | 8 | 3 | 64 |
| 64×64 | 8 | 4 | 2 | 16 | 2 | 16 |
| 256×256 | 16 | 16 | 0 | 16 | 4 | 256 |
| 256×256 | 16 | 8 | 4 | 32 | 3 | 64 |

A `ValueError` is raised at init if either constraint is violated.

---

## Loss functions

### Similarity

| Name | Notes |
|------|-------|
| `ncc` | Local NCC, 9×9 sliding window. Robust to intensity differences between sensors. Recommended for SAR. |
| `mse` | Mean squared error. Fast but sensitive to intensity scale. |
| `ssim` | Structural similarity. Good for same-modality pairs. |

### Regularization

| Name | Description |
|------|-------------|
| `deformation_gradient` | Mean squared Frobenius norm of the Jacobian of the displacement field, in pixel units. Naturally on the same scale as NCC — `reg_weight=0.1` is a reasonable starting point. **Recommended.** |
| `l2` | L2 penalty on spatial gradients of the field (first-order smoothness). |
| `bending` | Bending energy — second-order penalty, suppresses non-affine local deformations more aggressively. |

The `deformation_gradient` regularizer operates on `out["phi"]` — the **integrated** deformation field from the ODE, not the instantaneous velocity. This is more physically meaningful than regularizing the velocity alone.

```python
from src.training import RegistrationLoss

loss_fn = RegistrationLoss(
    similarity="ncc",
    regularization="deformation_gradient",
    similarity_weight=1.0,
    regularization_weight=0.1,
)

losses = loss_fn(out["warped"], fixed, out["phi"])
losses["total"]          # backward on this
losses["similarity"]     # detached, for logging
losses["regularization"] # detached, for logging
```

---

## Data augmentation

Augmentation is applied **only to training data**. Val and test sets are always unmodified so evaluation metrics are comparable.

### RandomRotation90

Rotates both images in a pair by the same random multiple of 90°. The same rotation is applied to both so the registration relationship is preserved. Square patches have no void corners after rotation.

### SARAwareRandomErasing

Erases a random rectangle **independently** in each image, simulating separate occlusion or interference in each SAR acquisition. The erased region is filled with Gaussian noise whose mean and std match the nonzero pixels of that specific patch, so the fill is statistically consistent with surrounding SAR content rather than being uniform random noise.

```
erasing_scale: [0.02, 0.2]   # erase 2%–20% of patch area
erasing_ratio: [0.3, 3.0]    # aspect ratio range of erased rectangle
```

### Expected train/val loss gap

Train loss will be **higher** than val loss — this is normal and expected:
1. Augmented training patches are harder (erasing degrades NCC scores)
2. Dropout is active during training but disabled during evaluation

Watch the **val loss trajectory** over epochs, not the absolute train/val gap. Decreasing val loss means the model is learning. Rising val loss with falling train loss indicates overfitting.

---

## Training scripts

| Script | Use case |
|--------|----------|
| `scripts/train_hybrid_patches.py` | SAR patch pairs from `patches.npy` |
| `scripts/train_numpy.py` | Individual `.npy` files with atlas |
| `scripts/train.py` | NIfTI / medical images |
| `scripts/evaluate.py` | Evaluate a checkpoint on the test set |
| `scripts/visualize_results.py` | Registration result figures |
| `scripts/ablation.py` | Hyperparameter sweep |
| `scripts/generate_dummy_data.py` | Generate atlas-based dummy data |

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint results/my_run/checkpoints/best.pt \
    --save-figures
```

Outputs in `results/<experiment_name>/`:
- `test_results.csv` — per-sample metrics
- `test_summary.csv` — mean ± std with bootstrapped 95% CIs
- `table.tex` — LaTeX-ready results table
- `figures/` — registration result images

---

## TensorBoard

```bash
python -m tensorboard.main --logdir results/
```

The **SCALARS** tab shows training/validation loss (total, similarity, regularization) and learning rate per epoch. The **HPARAMS** tab compares hyperparameters vs `best_val_loss` across all runs — useful for ablation studies.

---

## Adding a new model

1. Create `src/models/my_model.py`, subclassing `BaseRegistrationModel`
2. Implement `forward(moving, fixed)` returning a dict with at minimum:
   - `"warped"` — `(B, 1, H, W)` registered image
   - `"phi"` — `(B, 2, H, W)` deformation field in normalised `[-1, 1]` coordinates (required by `RegistrationLoss`)
3. Implement `get_config()` returning a dict of constructor kwargs
4. Register in `src/models/__init__.py`:
   ```python
   from .my_model import MyModel
   REGISTRY["MyModel"] = MyModel
   ```
5. Create `configs/experiments/my_model.yaml` with `model.name: "MyModel"`

---

## Project structure

```
imageRegistration/
├── scripts/
│   ├── train_hybrid_patches.py   SAR patch-pair training (primary)
│   ├── train_numpy.py            Atlas-based numpy training
│   ├── train.py                  NIfTI training
│   ├── evaluate.py               Test set evaluation
│   ├── visualize_results.py      Registration figures
│   ├── ablation.py               Hyperparameter sweep
│   └── generate_dummy_data.py    Dummy atlas data
├── src/
│   ├── data/
│   │   ├── patch_dataset.py      PatchPairDataset, SARAwareRandomErasing, RandomRotation90
│   │   ├── numpy_dataset.py      NumpyPairDataset, NumpyAtlasDataset
│   │   └── dataset.py            AtlasDataset, PairDataset (NIfTI)
│   ├── models/
│   │   ├── hybrid_ode_registration.py  HybridODERegistration (recommended)
│   │   ├── neural_ode_registration.py  NeuralODERegistration
│   │   ├── unet_registration.py        UNetRegistration
│   │   └── simple_cnn.py               SimpleCNN
│   ├── training/
│   │   ├── losses.py             NCC, MSE, SSIM, DeformationGradientLoss, RegistrationLoss
│   │   └── trainer.py            Trainer, build_optimizer, build_scheduler
│   ├── evaluation/
│   │   └── metrics.py            NCC, SSIM, TRE, Dice
│   └── utils/
│       ├── checkpointing.py
│       ├── logging.py            TensorBoard MetricLogger
│       └── visualization.py
├── CHEATSHEET.txt                Quick command reference
├── requirements.txt
└── README.md
```
