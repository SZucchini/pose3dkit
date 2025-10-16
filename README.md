# pose3dkit

Utilities for monocular 3D human pose estimation.

## Metrics

The MPJPE helpers live in `pose3dkit.metrics`.

- `compute_mpjpe(predicted, target, *, reduce_axes="all", keepdims=False, joint_mask=None, joint_weights=None)`
  - Accepts numpy arrays or torch tensors with shape `(B, T, J, 3)` or `(T, J, 3)`.
  - Returns values in the same backend/dtype/device as the inputs.
  - Choose the output layout by selecting axes to average over: e.g. `reduce_axes="none"` keeps `(B, T, J)`, `("joint",)` yields `(B, T)`, `("joint", "time")` yields `(B,)`, `("batch", "joint")` yields `(T,)`, and `"global"` collapses to a scalar.
  - Optional `joint_mask` (boolean) and `joint_weights` (non-negative) broadcast over `(B, T, J)` to skip or re-weight joints. Masked entries return zero when `reduce_axes="none"` and are ignored in reductions otherwise.
  - Inputs must be floating point and finite; NaNs/Infs raise `ValueError`.
- `compute_p_mpjpe(predicted, target, *, reduce_axes="all", keepdims=False)`
  - Procrustes-aligns (scale-rotation-translation) before measuring error and supports `(B, T, J, 3)` or `(T, J, 3)` inputs across numpy/torch backends.
  - The aligned prediction adopts the target’s coordinate scale, delivering a unit-consistent error regardless of the original magnitude of `predicted`.
  - Shares the same reduction semantics as `compute_mpjpe`; raises `ValueError` when the alignment is ill-defined (e.g., zero-variance sequences).

- `mpjpe_loss(predicted, target, *, reduction="mean", joint_mask=None, joint_weights=None)`
  - Torch-only functional loss helper that reuses `compute_mpjpe`.
  - Supports `reduction="mean"` (scalar) or `"none"` (per-sample `(B,)`, requires batch dimension).
- `p_mpjpe_loss(predicted, target, *, reduction="mean")`
  - Torch-only PA-MPJPE loss matching the original `sample.py` behaviour, supporting batch inputs and `reduction="mean"` / `"none"`.
  - Internally rescales `predicted` to the target’s unit system during the rigid alignment step.
- `n_mpjpe_loss(predicted, target, *, reduction="mean")`
  - Torch-only normalized MPJPE (scale-only) matching the historical `sample.py` implementation.
  - Supports `reduction="mean"` (scalar) or `"none"` (per-sample `(B,)`, requires batch dimension).
- `velocity_loss(predicted, target, *, reduction="mean")`
  - Torch-only mean per-joint velocity error mirroring the historical `sample.py` helper.
  - Supports `reduction="mean"` (scalar) or `"none"` (per-sample `(B,)`, requires batch dimension).

- `MPJPELoss(reduction="mean", joint_mask=None, joint_weights=None)`
  - Thin `torch.nn.Module` wrapper around `mpjpe_loss`.
- `PMPJPELoss(reduction="mean")`
  - `torch.nn.Module` wrapper around `p_mpjpe_loss`.
- `NMPJPELoss(reduction="mean")`
  - `torch.nn.Module` wrapper around `n_mpjpe_loss`.
- `VelocityLoss(reduction="mean")`
  - `torch.nn.Module` wrapper around `velocity_loss`.

- `pose3dkit.MPJPE()`
  - TorchMetrics-compatible metric returning a scalar global MPJPE from `(B, T, J, 3)` tensors.
  - Works with DataParallel and DDP thanks to `dist_reduce_fx="sum"`, and preserves mixed-precision dtypes in the final result.
- `pose3dkit.PMPJPE()`
  - TorchMetrics-compatible PA-MPJPE counterpart with the same DDP-friendly accumulation strategy.

Units are not enforced—ensure callers use consistent coordinate scales (e.g. millimetres vs metres).
