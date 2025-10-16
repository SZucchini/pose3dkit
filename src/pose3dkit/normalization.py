"""Utilities for normalizing and denormalizing keypoint coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _coerce_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _coerce_keypoints(kpts: Any, *, copy: bool) -> np.ndarray:
    array = np.array(kpts, dtype=float, copy=copy)
    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError("Keypoints array must have at least one coordinate dimension")
    return array


def _broadcast_parameter(
    value: Any, target_shape: tuple[int, ...], *, name: str
) -> np.ndarray:
    array = _coerce_array(value, name=name)
    if array.shape != target_shape:
        if array.ndim > len(target_shape):
            raise ValueError(
                f"{name} with shape {array.shape} cannot be broadcast to target shape {target_shape}"
            )
        expand_dims = len(target_shape) - array.ndim
        if expand_dims:
            array = np.reshape(array, array.shape + (1,) * expand_dims)
    try:
        return np.broadcast_to(array, target_shape)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            f"{name} with shape {array.shape} cannot be broadcast to target shape {target_shape}"
        ) from exc


def normalize_keypoints(
    kpts: Any,
    width: Any,
    height: Any,
    *,
    spatial_dims: int = 2,
    copy: bool = True,
) -> np.ndarray:
    """Normalize 2D or 3D keypoints to the range [-1, 1].

    The normalization matches the convention used in the provided reference code:

    * ``x`` and ``y`` coordinates are divided by the image width before mapping to
      the range [-1, 1] and [-height/width, height/width] respectively.
    * ``z`` coordinates (when present) are scaled using the image width.

    Args:
        kpts: Array-like keypoints with shape ``(..., J, C)`` where the last axis
            stores the coordinate values.
        width: Image width. Can be a scalar or an array broadcastable to the
            keypoints shape without the coordinate dimension.
        height: Image height. Must be broadcastable to the keypoints shape
            without the coordinate dimension.
        spatial_dims: Number of leading coordinate components to normalize. Set
            to ``2`` for 2D keypoints (even if an additional confidence channel
            is present) and ``3`` for 3D keypoints.
        copy: If ``True`` (default), a new array is returned. If ``False``, the
            input array is modified in-place when possible.

    Returns:
        np.ndarray: Normalized keypoints.

    Raises:
        ValueError: If the inputs cannot be broadcast together or when
        ``spatial_dims`` is not supported.

    """
    coords = _coerce_keypoints(kpts, copy=copy)
    if spatial_dims not in (2, 3):
        raise ValueError("spatial_dims must be either 2 or 3")
    if coords.shape[-1] < spatial_dims:
        raise ValueError(
            "Keypoints array does not contain enough coordinate dimensions for the requested spatial_dims"
        )

    width_b = _broadcast_parameter(width, coords.shape[:-1], name="width")
    height_b = _broadcast_parameter(height, coords.shape[:-1], name="height")

    if np.any(width_b == 0):
        raise ValueError("width must be non-zero")

    coords_x = coords[..., 0]
    coords[..., 0] = coords_x / width_b * 2.0 - 1.0

    coords_y = coords[..., 1]
    coords[..., 1] = coords_y / width_b * 2.0 - height_b / width_b

    if spatial_dims == 3:
        coords[..., 2] = coords[..., 2] / width_b * 2.0

    return coords


def denormalize_keypoints(
    kpts: Any,
    width: Any,
    height: Any,
    *,
    spatial_dims: int = 2,
    copy: bool = True,
) -> np.ndarray:
    """Denormalize keypoints that were previously normalized."""
    coords = _coerce_keypoints(kpts, copy=copy)
    if spatial_dims not in (2, 3):
        raise ValueError("spatial_dims must be either 2 or 3")
    if coords.shape[-1] < spatial_dims:
        raise ValueError(
            "Keypoints array does not contain enough coordinate dimensions for the requested spatial_dims"
        )

    width_b = _broadcast_parameter(width, coords.shape[:-1], name="width")
    height_b = _broadcast_parameter(height, coords.shape[:-1], name="height")

    if np.any(width_b == 0):
        raise ValueError("width must be non-zero")

    coords_x = coords[..., 0]
    coords[..., 0] = (coords_x + 1.0) * width_b / 2.0

    coords_y = coords[..., 1]
    coords[..., 1] = (coords_y + height_b / width_b) * width_b / 2.0

    if spatial_dims == 3:
        coords[..., 2] = coords[..., 2] * width_b / 2.0

    return coords


def normalize_keypoints_2d(
    kpts: Any,
    width: Any,
    height: Any,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Normalize 2D keypoints while preserving optional confidence channels."""
    return normalize_keypoints(
        kpts,
        width,
        height,
        spatial_dims=2,
        copy=copy,
    )


def denormalize_keypoints_2d(
    kpts: Any,
    width: Any,
    height: Any,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Denormalize 2D keypoints that were normalized with :func:`normalize_keypoints_2d`."""
    return denormalize_keypoints(
        kpts,
        width,
        height,
        spatial_dims=2,
        copy=copy,
    )


def normalize_keypoints_3d(
    kpts: Any,
    width: Any,
    height: Any,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Normalize 3D keypoints."""
    return normalize_keypoints(
        kpts,
        width,
        height,
        spatial_dims=3,
        copy=copy,
    )


def denormalize_keypoints_3d(
    kpts: Any,
    width: Any,
    height: Any,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Denormalize 3D keypoints that were normalized with :func:`normalize_keypoints_3d`."""
    return denormalize_keypoints(
        kpts,
        width,
        height,
        spatial_dims=3,
        copy=copy,
    )


__all__ = [
    "normalize_keypoints",
    "denormalize_keypoints",
    "normalize_keypoints_2d",
    "denormalize_keypoints_2d",
    "normalize_keypoints_3d",
    "denormalize_keypoints_3d",
]
