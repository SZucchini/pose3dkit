import numpy as np

from pose3dkit.normalization import (
    denormalize_keypoints,
    denormalize_keypoints_2d,
    denormalize_keypoints_3d,
    normalize_keypoints,
    normalize_keypoints_2d,
    normalize_keypoints_3d,
)


def test_normalize_denormalize_2d_with_confidence():
    kpts = np.array(
        [
            [[10.0, 20.0, 0.9], [100.0, 200.0, 0.8]],
            [[50.0, 400.0, 0.7], [1900.0, 1070.0, 0.6]],
        ]
    )
    width, height = 1920.0, 1080.0

    normalized = normalize_keypoints_2d(kpts, width, height)

    expected_xy = kpts[..., :2] / width * 2 - np.array([1.0, height / width])
    np.testing.assert_allclose(normalized[..., :2], expected_xy)
    np.testing.assert_allclose(normalized[..., 2], kpts[..., 2])

    restored = denormalize_keypoints_2d(normalized, width, height)
    np.testing.assert_allclose(restored, kpts)


def test_normalize_denormalize_3d_with_per_sample_resolution():
    kpts = np.array(
        [
            [[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
            [[40.0, 80.0, 120.0], [60.0, 90.0, 150.0]],
        ]
    )
    width = np.array([1920.0, 1280.0])
    height = np.array([1080.0, 720.0])

    normalized = normalize_keypoints_3d(kpts, width, height)

    width_b = np.broadcast_to(width[:, None], kpts.shape[:2])
    height_b = np.broadcast_to(height[:, None], kpts.shape[:2])
    offset = np.stack((np.ones_like(width_b), height_b / width_b), axis=-1)
    expected_xy = kpts[..., :2] / width_b[..., None] * 2 - offset
    expected_z = kpts[..., 2] / width_b * 2

    np.testing.assert_allclose(normalized[..., :2], expected_xy)
    np.testing.assert_allclose(normalized[..., 2], expected_z)

    restored = denormalize_keypoints_3d(normalized, width, height)
    np.testing.assert_allclose(restored, kpts)


def test_general_helpers_raise_for_invalid_spatial_dims():
    kpts = np.zeros((2, 3))
    width, height = 1920.0, 1080.0

    try:
        normalize_keypoints(kpts, width, height, spatial_dims=4)
    except ValueError as exc:
        assert "spatial_dims" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError for unsupported spatial_dims")

    try:
        denormalize_keypoints(kpts, width, height, spatial_dims=0)
    except ValueError as exc:
        assert "spatial_dims" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError for unsupported spatial_dims")
