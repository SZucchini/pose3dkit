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

- `mpjpe_loss(predicted, target, *, reduction="mean", joint_mask=None, joint_weights=None)`
  - Torch-only functional loss helper that reuses `compute_mpjpe`.
  - Supports `reduction="mean"` (scalar) or `"none"` (per-sample `(B,)`, requires batch dimension).

- `MPJPELoss(reduction="mean", joint_mask=None, joint_weights=None)`
  - Thin `torch.nn.Module` wrapper around `mpjpe_loss`.

- `pose3dkit.MPJPE()`
  - TorchMetrics-compatible metric returning a scalar global MPJPE from `(B, T, J, 3)` tensors.
  - Works with DataParallel and DDP thanks to `dist_reduce_fx="sum"`, and preserves mixed-precision dtypes in the final result.

Units are not enforced—ensure callers use consistent coordinate scales (e.g. millimetres vs metres).
