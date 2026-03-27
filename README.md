# Image Registration

A PyTorch framework for 2D image registration. Designed for SAR image pairs from [sar_align](../sar_align) but works with any `(N, 2, H, W)` patch file.

---

## How it works

### GateFlow (recommended)

Iterative Lagrangian registration. At each of `n_t` steps the network predicts an incremental displacement `delta` from the current warped image and the fixed target. Displacements are accumulated additively — valid because each delta is kept small (`delta / n_t`) so addition approximates composition.

```
phi_0  = 0
for i in 1..n_t:
    delta_i  = velocity_net( warp(moving, phi_{i-1}), fixed )  / n_t
    phi_i    = phi_{i-1} + delta_i
warped = warp(moving, phi_{n_t})
```

### Velocity network architecture

```
[f_current, fixed]
    → Stem (L2NormConv2d, separate per image)   (B, 1, H, W) → (B, 2C, H', W')
    → Gate:  prod = conv_f(f) * conv_g(g)
             gate = prod[:,:C] + prod[:,C:]      fold → (B, C, H', W')
    → Bottleneck: expand( tanh( squeeze(gate) ) )
    → PE branch: fixed sinusoidal PE modulated by gate → velocity residual
    → Bilinear upsample → (B, 2, H, W)
```

The **L2NormConv2d** stem normalises every filter to unit norm before each forward pass, making it analogous to normalised cross-correlation.

The **PE residual branch** projects the sinusoidal positional encoding through the gate, then subtracts the original PE — the network predicts the *change* in positional encoding, which drives the displacement head.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick start

```bash
# Generate synthetic patches for testing
python scripts/generate_synthetic_patches.py

# Train GateFlow on synthetic patches
python scripts/train_image_pairs.py --config configs/experiments/gate_flow_synthetic.yaml

# Train on real SAR patches
python scripts/train_image_pairs.py \
    --config configs/experiments/gate_flow_synthetic.yaml \
    --patches path/to/patches.npy
```

---

## Loading your own data

The data pipeline expects a single `patches.npy` file of shape `(N, 2, H, W)` float32 with values in `[0, 1]`. Channel 0 is the moving image, channel 1 is the fixed image.

```python
from src.data import build_patch_dataloaders

cfg = {
    "data": {
        "patches_path": "dataset/synthetic_patches.npy",
        "val_frac":  0.15,
        "test_frac": 0.15,
        "seed": 42,
        "num_workers": 0,
    },
    "training": {"batch_size": 16},
    "augmentation": {
        "use_rotation": True,
        "use_erasing":  True,
    },
}
train_loader, val_loader, test_loader = build_patch_dataloaders(cfg)
```

The file is memory-mapped (`mmap_mode='r'`) so large datasets are never fully loaded into RAM.

---

## Models

| Model | Description |
|-------|-------------|
| `GateFlow` | Iterative Lagrangian registration with gated CNN stem and PE residual. **Recommended.** |
| `SimpleCNN` | Minimal CNN baseline |
| `NeuralODERegistration` | Transport ODE with CNN velocity network |
| `HybridODERegistration` | Transport ODE with Gated Stem + Transformer; integrates deformation field jointly |
| `EulerianHybridODERegistration` | Eulerian variant of the hybrid ODE |
| `DirectHybridRegistration` | Direct (non-ODE) hybrid registration |
| `UNetRegistration` | Full UNet encoder-decoder, 2D/3D capable |

All models return a dict from `forward(moving, fixed)`.

### GateFlow output dict

```python
out = model(moving, fixed)
out["warped"]      # (B, 1, H, W)          registered image
out["phi"]         # (B, 2, H, W)          accumulated displacement in [-1, 1] coords
out["flow"]        # (B, 2, H, W)          same as phi (alias)
out["trajectory"]  # (n_t+1, B, 1, H, W)  intermediate states from t=0 to t=1
```

### GateFlow config

```yaml
model:
  name: GateFlow
  in_channels:         1
  hidden_channels:     256   # C — gate width
  kernel_size:         8     # k — square stem conv kernel
  stride:              4     # s — H' = H/s
  bottleneck_channels: 256   # K — tanh bottleneck width
  n_pe:                256   # PE channels (must be divisible by 4)
  n_t:                 20    # number of iterative refinement steps
```

---

## Loss functions

### Similarity

| Name | Notes |
|------|-------|
| `ncc` | Local NCC, sliding window. Robust to intensity differences. Recommended for SAR. |
| `mse` | Mean squared error. |
| `ssim` | Structural similarity. |

### Regularization

| Name | Description |
|------|-------------|
| `l2` | L2 penalty on spatial gradients of the displacement field. |
| `bending` | Bending energy — second-order penalty. |
| `deformation_gradient` | Mean squared Frobenius norm of the displacement Jacobian, in pixel units. |

### Jacobian regularization (optional)

| Key | Description |
|-----|-------------|
| `jacobian_det_weight` | Penalises folds: `relu(-(det(J) + eps))^2`. Set `> 0` to enable. |
| `jacobian_eps` | Margin — penalises `det(J) < -eps` (default `1e-3`). |
| `log_jacobian_weight` | Volume preservation: `(log det(J))^2`. Set `> 0` to enable. |
| `log_jacobian_clamp` | Clamp `det` before log to avoid `log(0)` (default `1e-5`). |

```yaml
loss:
  similarity:            ncc
  regularization:        l2
  similarity_weight:     1.0
  regularization_weight: 0.1
  ncc_win:               9
  jacobian_det_weight:   0.1   # 0 = disabled
  jacobian_eps:          0.001
  log_jacobian_weight:   0.1   # 0 = disabled
  log_jacobian_clamp:    0.00001
```

> **Note:** Write Jacobian float values as decimals (`0.001`) rather than scientific notation (`1e-3`) in YAML, or they will be parsed as strings.

```python
from src.training import RegistrationLoss

loss_fn = RegistrationLoss(
    similarity="ncc",
    regularization="l2",
    similarity_weight=1.0,
    regularization_weight=0.1,
    jacobian_det_weight=0.1,
    log_jacobian_weight=0.1,
)

losses = loss_fn(out["warped"], fixed, out["phi"])
losses["total"]          # backward on this
losses["similarity"]     # detached, for logging
losses["regularization"] # detached, for logging
losses["jacobian"]       # detached, for logging
losses["log_jacobian"]   # detached, for logging
```

---

## Data augmentation

Applied only to training data. Val and test sets are always unmodified.

### RandomRotation90

Rotates both images in a pair by the same random multiple of 90°. Preserves the registration relationship. Square patches have no void corners after rotation.

### SARAwareRandomErasing

Erases a random rectangle **independently** in each image, simulating separate occlusion in each acquisition. Fill values are drawn from `N(patch_mean, patch_std)` of nonzero pixels, making the fill statistically consistent with surrounding SAR content.

```yaml
augmentation:
  use_rotation: false
  use_erasing:  false
```

> Train loss will be **higher** than val loss when augmentation is active — this is expected. Watch the val loss trajectory, not the absolute gap.

---

## Training

All training goes through `train_image_pairs.py`. CLI flags override the config when provided.

```bash
# Basic
python scripts/train_image_pairs.py --config configs/experiments/gate_flow_synthetic.yaml

# Override patches path
python scripts/train_image_pairs.py \
    --config configs/experiments/gate_flow_synthetic.yaml \
    --patches path/to/patches.npy

# Override training settings
python scripts/train_image_pairs.py \
    --config configs/experiments/gate_flow_synthetic.yaml \
    --epochs 50 --batch-size 32 --reg-weight 0.05

# Initialise weights from a prior checkpoint (optimizer and epoch are NOT restored)
python scripts/train_image_pairs.py \
    --config configs/experiments/gate_flow_synthetic.yaml \
    --weights outputs/gate_flow_synthetic/checkpoints/best.pt

# Disable early stopping
python scripts/train_image_pairs.py \
    --config configs/experiments/gate_flow_synthetic.yaml \
    --patience 0
```

Outputs are written to `<output_dir>/<experiment_name>/`:

```
outputs/<experiment_name>/
    config.yaml           copy of config used
    checkpoints/
        best.pt           best validation checkpoint
        latest.pt         most recent checkpoint
```

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/gate_flow_synthetic/checkpoints/best.pt \
    --save-figures
```

Outputs in `outputs/<experiment_name>/`:
- `test_results.csv` — per-sample metrics
- `test_summary.csv` — mean ± std with bootstrapped 95% CIs
- `table.tex` — LaTeX-ready results table
- `figures/` — registration result images

---

## Visualization

### Registration results

```bash
python scripts/visualize_results.py \
    --checkpoint outputs/gate_flow_synthetic/checkpoints/best.pt \
    --n-samples 8 \
    --out figures/results.png
```

Shows a grid with four columns per sample:
1. Moving image
2. Fixed image
3. Warped (registered) image
4. Deformed grid — how the underlying coordinate grid is displaced by `phi`

Grid output is also saved separately as `*_grid.png`.

### Learned filter pairs

```bash
python scripts/visualize_filters.py \
    --checkpoint outputs/gate_flow_synthetic/checkpoints/best.pt \
    --n-pairs 25 \
    --out figures/filters.png
```

Shows filter quads from `conv_f` and `conv_g` in the GateFlow stem. Each quad shows:
```
conv_f[c]    conv_g[c]       ← phase A
conv_f[c+C]  conv_g[c+C]    ← phase B (90° shifted for ISA-style filters)
```

Well-trained filters should exhibit Gabor-like quadrature pairs at matching orientations/frequencies.

---

## TensorBoard

```bash
python -m tensorboard.main --logdir outputs/
```

The **SCALARS** tab shows training/validation loss (total, similarity, regularization, jacobian, log_jacobian) and learning rate per epoch. The **HPARAMS** tab compares hyperparameters vs `best_val_loss` across runs.

---

## Ablation study

Add an `ablation` section to any experiment config:

```yaml
ablation:
  param:  model.hidden_channels   # dot-path to the parameter
  values: [64, 128, 256, 512]
```

Then run:

```bash
python scripts/ablation.py --config configs/experiments/ablation_features.yaml
```

Outputs a comparison CSV and figure in `outputs/<experiment_name>/`.

---

## Adding a new model

1. Create `src/models/my_model.py`, subclassing `BaseRegistrationModel`
2. Implement `forward(moving, fixed)` returning a dict with at minimum:
   - `"warped"` — `(B, 1, H, W)` registered image
   - `"phi"` — `(B, 2, H, W)` deformation field in normalised `[-1, 1]` coordinates
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
│   ├── train_image_pairs.py        Config-driven training (all models)
│   ├── evaluate.py                 Test set evaluation
│   ├── visualize_results.py        Registration figures + deformed grid
│   ├── visualize_filters.py        GateFlow stem filter visualisation
│   ├── ablation.py                 Hyperparameter sweep
│   ├── generate_synthetic_patches.py  Synthetic shape patch generator
│   └── inspect_patches.py          Dataset inspection utility
├── src/
│   ├── data/
│   │   └── patch_dataset.py        PatchPairDataset, build_patch_dataloaders,
│   │                               RandomRotation90, SARAwareRandomErasing
│   ├── models/
│   │   ├── gate_flow.py            GateFlow (recommended)
│   │   ├── hybrid_ode_registration.py
│   │   ├── eulerian_hybrid_registration.py
│   │   ├── direct_hybrid_registration.py
│   │   ├── neural_ode_registration.py
│   │   ├── unet_registration.py
│   │   └── simple_cnn.py
│   ├── training/
│   │   ├── losses.py               NCC, MSE, SSIM, GradientSmoothnessLoss,
│   │   │                           BendingEnergyLoss, NegativeJacobianLoss,
│   │   │                           LogJacobianLoss, RegistrationLoss
│   │   └── trainer.py              Trainer, build_optimizer, build_scheduler
│   ├── evaluation/
│   │   └── metrics.py              NCC, SSIM, TRE, Dice
│   └── utils/
│       ├── checkpointing.py
│       ├── logging.py              TensorBoard MetricLogger
│       └── visualization.py
├── configs/experiments/
│   ├── gate_flow_synthetic.yaml    GateFlow on synthetic patches (primary)
│   ├── direct_hybrid_synthetic.yaml
│   ├── direct_hybrid_sar.yaml
│   └── ...
├── dataset/
│   └── synthetic_patches.npy       Generated by generate_synthetic_patches.py
├── outputs/                        Training outputs (checkpoints, logs)
├── CHEATSHEET.txt                  Quick command reference
├── requirements.txt
└── README.md
```
