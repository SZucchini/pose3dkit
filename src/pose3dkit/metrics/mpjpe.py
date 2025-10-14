from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]

ArrayLike = Union[np.ndarray, "torch.Tensor"]
ReduceAxes = Union[str, Iterable[str], None]

_VALID_AXIS_NAMES = ("batch", "time", "joint")


@dataclass(frozen=True)
class _ShapeInfo:
    shape: Tuple[int, ...]
    axis_names: Tuple[str, ...]

    @property
    def has_batch(self) -> bool:
        return "batch" in self.axis_names


def compute_mpjpe(
    predicted: ArrayLike,
    target: ArrayLike,
    *,
    reduce_axes: ReduceAxes = "all",
    keepdims: bool = False,
    joint_mask: ArrayLike | None = None,
    joint_weights: ArrayLike | None = None,
) -> ArrayLike:
    """Compute Mean Per-Joint Position Error for numpy arrays or torch tensors.

    Args:
        predicted: Predicted keypoints. Shape must be (B, T, J, 3) or (T, J, 3).
        target: Target keypoints. Same type and shape as ``predicted``.
        reduce_axes: Axes to average over. Accepts ``"all"``/``"global"``,
            ``"none"``, any single axis name (``"batch"``, ``"time"``, ``"joint"``),
            or an iterable of axis names. Defaults to all available axes.
        keepdims: If ``True`` keeps reduced dimensions with size 1.
        joint_mask: Optional boolean mask broadcastable to the per-joint error shape
            (B, T, J) or (T, J). Masked positions are ignored in reductions; if
            ``reduce_axes="none"`` masked entries are returned as zero.
        joint_weights: Optional non-negative weights broadcastable to (B, T, J) or
            (T, J). Applied multiplicatively before reductions.

    Returns:
        Array/tensor of MPJPE values with the same backend, dtype, and device as the
        inputs.

    Raises:
        ValueError: If shapes, dtypes, masks, or reduction axes are invalid, or if
            data contain NaNs/Infs, or if torch operations are requested without
            torch installed.

    """
    backend = _detect_backend(predicted, target)
    shape_info = _validate_inputs(predicted, target, backend)
    per_joint_error = _pairwise_distance(predicted, target, backend)

    weights = _prepare_weights(
        per_joint_error,
        backend=backend,
        joint_mask=joint_mask,
        joint_weights=joint_weights,
    )

    axes_to_reduce = _normalize_reduce_axes(reduce_axes, shape_info)

    if not axes_to_reduce:
        if weights is not None:
            mask = weights > 0
            if backend == "torch":
                assert torch is not None
                per_joint_error = per_joint_error * mask.to(per_joint_error.dtype)
            else:
                per_joint_error = per_joint_error * mask.astype(per_joint_error.dtype, copy=False)
        return per_joint_error

    axis_indices = tuple(sorted(shape_info.axis_names.index(axis) for axis in axes_to_reduce))
    if backend == "torch":
        return _reduce_torch(per_joint_error, weights, axis_indices, keepdims)
    return _reduce_numpy(per_joint_error, weights, axis_indices, keepdims)


def mpjpe_loss(
    predicted: "torch.Tensor",
    target: "torch.Tensor",
    *,
    reduction: str = "mean",
    joint_mask: "torch.Tensor | None" = None,
    joint_weights: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Torch loss helper returning MPJPE suitable for backpropagation.

    Args:
        predicted: Predicted keypoints with shape (B, T, J, 3) or (T, J, 3).
        target: Target keypoints with identical shape.
        reduction: Either ``"mean"`` (default) or ``"none"``. ``"none"``
            returns per-sample values with shape (B,) when batch axis is present.
        joint_mask: Optional boolean mask broadcastable to (B, T, J) or (T, J).
        joint_weights: Optional weights broadcastable to (B, T, J) or (T, J).

    Returns:
        Torch tensor containing the reduced loss.

    """
    if torch is None:
        raise RuntimeError("torch is required to compute mpjpe_loss.")
    if not isinstance(predicted, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("mpjpe_loss requires torch.Tensor inputs.")

    has_batch = predicted.ndim == 4
    if reduction not in {"mean", "none"}:
        raise ValueError("Supported reductions are 'mean' and 'none'.")
    if reduction == "none" and not has_batch:
        raise ValueError("reduction='none' requires a batch dimension.")

    if reduction == "none":
        axes: ReduceAxes = ("time", "joint")
        keepdims = False
    else:  # "mean"
        axes = ("batch", "time", "joint") if has_batch else ("time", "joint")
        keepdims = False

    return compute_mpjpe(
        predicted,
        target,
        reduce_axes=axes,
        keepdims=keepdims,
        joint_mask=joint_mask,
        joint_weights=joint_weights,
    )


if torch is not None:

    class MPJPELoss(torch.nn.Module):
        """Torch module wrapper around :func:`mpjpe_loss`."""

        def __init__(
            self,
            *,
            reduction: str = "mean",
            joint_mask: "torch.Tensor | None" = None,
            joint_weights: "torch.Tensor | None" = None,
        ) -> None:
            super().__init__()
            self.reduction = reduction
            if joint_mask is not None:
                self.register_buffer("joint_mask", joint_mask, persistent=False)
            else:
                self.register_buffer("joint_mask", None, persistent=False)
            if joint_weights is not None:
                self.register_buffer("joint_weights", joint_weights, persistent=False)
            else:
                self.register_buffer("joint_weights", None, persistent=False)

        def forward(self, predicted: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
            return mpjpe_loss(
                predicted,
                target,
                reduction=self.reduction,
                joint_mask=self.joint_mask,
                joint_weights=self.joint_weights,
            )

else:

    class MPJPELoss:  # type: ignore[too-many-ancestors]
        """Placeholder raising an informative error when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - torch missing
            raise RuntimeError("torch is required to use MPJPELoss.")

        def forward(self, predicted, target):  # pragma: no cover - torch missing
            raise RuntimeError("torch is required to use MPJPELoss.")


def _detect_backend(predicted: ArrayLike, target: ArrayLike) -> str:
    if torch is not None and isinstance(predicted, torch.Tensor):
        if not isinstance(target, torch.Tensor):
            raise TypeError("Predicted and target must share the same backend.")
        return "torch"
    if isinstance(predicted, np.ndarray):
        if not isinstance(target, np.ndarray):
            raise TypeError("Predicted and target must share the same backend.")
        return "numpy"
    raise TypeError("Predicted and target must be numpy.ndarray or torch.Tensor.")


def _validate_inputs(predicted: ArrayLike, target: ArrayLike, backend: str) -> _ShapeInfo:
    shape = tuple(predicted.shape)
    if shape != tuple(target.shape):
        raise ValueError(f"Predicted shape {shape} must match target shape {target.shape}.")
    if len(shape) not in (3, 4):
        raise ValueError("Expected shape (B, T, J, 3) or (T, J, 3).")
    if shape[-1] != 3:
        raise ValueError("Final dimension must be 3 (x, y, z).")
    if backend == "torch":
        assert isinstance(predicted, torch.Tensor)
        if not predicted.dtype.is_floating_point:
            raise TypeError("Predicted tensor must have a floating-point dtype.")
        if not target.dtype.is_floating_point:
            raise TypeError("Target tensor must have a floating-point dtype.")
        if not torch.isfinite(predicted).all():
            raise ValueError("Predicted tensor contains NaN or Inf values.")
        if not torch.isfinite(target).all():
            raise ValueError("Target tensor contains NaN or Inf values.")
    else:
        assert isinstance(predicted, np.ndarray)
        if not np.issubdtype(predicted.dtype, np.floating):
            raise TypeError("Predicted array must have a floating-point dtype.")
        if not np.issubdtype(target.dtype, np.floating):
            raise TypeError("Target array must have a floating-point dtype.")
        if not np.isfinite(predicted).all():
            raise ValueError("Predicted array contains NaN or Inf values.")
        if not np.isfinite(target).all():
            raise ValueError("Target array contains NaN or Inf values.")

    axis_names: Tuple[str, ...]
    if len(shape) == 4:
        axis_names = ("batch", "time", "joint")
    else:
        axis_names = ("time", "joint")
    return _ShapeInfo(shape=shape[:-1], axis_names=axis_names)


def _pairwise_distance(predicted: ArrayLike, target: ArrayLike, backend: str) -> ArrayLike:
    if backend == "torch":
        diff = predicted - target
        squared = diff.mul(diff)
        summed = squared.sum(dim=-1)
        return summed.clamp(min=0).sqrt()
    diff = predicted - target
    squared = np.square(diff, dtype=diff.dtype)
    summed = np.sum(squared, axis=-1, dtype=diff.dtype)
    np.sqrt(summed, out=summed)
    return summed


def _prepare_weights(
    errors: ArrayLike,
    *,
    backend: str,
    joint_mask: ArrayLike | None,
    joint_weights: ArrayLike | None,
) -> ArrayLike | None:
    if joint_mask is None and joint_weights is None:
        return None

    if backend == "torch":
        assert torch is not None
        base = torch.ones_like(errors, dtype=errors.dtype, device=errors.device)
        if joint_mask is not None:
            mask_tensor = _to_torch_broadcastable(
                joint_mask,
                target_shape=base.shape,
                dtype=torch.bool,
                device=base.device,
                name="joint_mask",
            )
            base = base * mask_tensor.to(base.dtype)
        if joint_weights is not None:
            weights_tensor = _to_torch_broadcastable(
                joint_weights,
                target_shape=base.shape,
                dtype=base.dtype,
                device=base.device,
                name="joint_weights",
            )
            if (weights_tensor < 0).any():
                raise ValueError("joint_weights must be non-negative.")
            base = base * weights_tensor
        return base

    base_np = np.ones_like(errors, dtype=errors.dtype)
    if joint_mask is not None:
        mask_array = _to_numpy_broadcastable(
            joint_mask,
            target_shape=base_np.shape,
            dtype=bool,
            name="joint_mask",
        )
        base_np = base_np * mask_array.astype(base_np.dtype, copy=False)
    if joint_weights is not None:
        weights_array = _to_numpy_broadcastable(
            joint_weights,
            target_shape=base_np.shape,
            dtype=base_np.dtype,
            name="joint_weights",
        )
        if np.any(weights_array < 0):
            raise ValueError("joint_weights must be non-negative.")
        base_np = base_np * weights_array.astype(base_np.dtype, copy=False)
    return base_np


def _normalize_reduce_axes(reduce_axes: ReduceAxes, shape_info: _ShapeInfo) -> Tuple[str, ...]:
    if reduce_axes is None:
        reduce_axes = "all"
    if isinstance(reduce_axes, str):
        key = reduce_axes.lower()
        if key in ("all", "global", "mean"):
            axes = shape_info.axis_names
        elif key == "none":
            axes = ()
        else:
            axes = tuple(filter(None, key.split("_")))
    else:
        axes = tuple(reduce_axes)

    for axis in axes:
        if axis not in _VALID_AXIS_NAMES:
            raise ValueError(f"Unknown axis '{axis}'. Valid axes are {_VALID_AXIS_NAMES}.")
        if axis == "batch" and not shape_info.has_batch:
            raise ValueError("Batch axis is not available for (T, J, 3) inputs.")
        if axis not in shape_info.axis_names:
            raise ValueError(f"Axis '{axis}' is not present in the current input shape.")

    seen = []
    for axis in axes:
        if axis not in seen:
            seen.append(axis)
    return tuple(seen)


def _reduce_torch(
    errors: "torch.Tensor",
    weights: "torch.Tensor | None",
    axis_indices: Tuple[int, ...],
    keepdims: bool,
) -> "torch.Tensor":
    assert torch is not None
    if not axis_indices:
        return errors

    if weights is None:
        total = errors.sum(dim=axis_indices, keepdim=keepdims)
        count = _torch_count(errors, axis_indices, keepdims)
    else:
        total = (errors * weights).sum(dim=axis_indices, keepdim=keepdims)
        count = weights.sum(dim=axis_indices, keepdim=keepdims)

    if torch.any(count == 0):
        raise ValueError("Reduction encountered zero total weight; check joint_mask/joint_weights.")
    return total / count


def _reduce_numpy(
    errors: np.ndarray,
    weights: np.ndarray | None,
    axis_indices: Tuple[int, ...],
    keepdims: bool,
) -> np.ndarray:
    if not axis_indices:
        return errors

    if weights is None:
        total = np.sum(errors, axis=axis_indices, keepdims=keepdims, dtype=errors.dtype)
        count = _numpy_count(errors, axis_indices, keepdims)
    else:
        total = np.sum(errors * weights, axis=axis_indices, keepdims=keepdims, dtype=errors.dtype)
        count = np.sum(weights, axis=axis_indices, keepdims=keepdims, dtype=errors.dtype)

    if np.any(count == 0):
        raise ValueError("Reduction encountered zero total weight; check joint_mask/joint_weights.")
    return total / count


def _torch_count(tensor: "torch.Tensor", axis_indices: Tuple[int, ...], keepdims: bool) -> "torch.Tensor":
    count = 1
    for axis in axis_indices:
        count *= tensor.shape[axis]
    count_tensor = torch.tensor(count, dtype=tensor.dtype, device=tensor.device)
    if keepdims:
        shape = list(tensor.shape)
        for axis in axis_indices:
            shape[axis] = 1
        return count_tensor.view([1] * len(shape)).expand(shape)
    return count_tensor


def _numpy_count(array: np.ndarray, axis_indices: Tuple[int, ...], keepdims: bool) -> np.ndarray:
    count = 1
    for axis in axis_indices:
        count *= array.shape[axis]
    if keepdims:
        shape = list(array.shape)
        for axis in axis_indices:
            shape[axis] = 1
        return np.full(shape, count, dtype=array.dtype)
    return np.array(count, dtype=array.dtype)


def _to_torch_broadcastable(
    value: ArrayLike,
    *,
    target_shape: Tuple[int, ...],
    dtype: "torch.dtype",
    device: "torch.device",
    name: str,
) -> "torch.Tensor":
    assert torch is not None
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim > len(target_shape):
        raise ValueError(f"{name} has too many dimensions to broadcast to {target_shape}.")
    view_shape = (1,) * (len(target_shape) - tensor.ndim) + tensor.shape
    tensor = tensor.reshape(view_shape)
    try:
        return tensor.expand(target_shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} with shape {tensor.shape} cannot broadcast to {target_shape}.") from exc


def _to_numpy_broadcastable(
    value: ArrayLike,
    *,
    target_shape: Tuple[int, ...],
    dtype,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim > len(target_shape):
        raise ValueError(f"{name} has too many dimensions to broadcast to {target_shape}.")
    view_shape = (1,) * (len(target_shape) - array.ndim) + array.shape
    array = array.reshape(view_shape)
    try:
        return np.broadcast_to(array, target_shape)
    except ValueError as exc:
        raise ValueError(f"{name} with shape {array.shape} cannot broadcast to {target_shape}.") from exc


__all__ = ["compute_mpjpe", "mpjpe_loss", "MPJPELoss"]
