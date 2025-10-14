import numpy as np
import pytest

from pose3dkit import MPJPE
from pose3dkit.metrics import MPJPELoss, compute_mpjpe, mpjpe_loss


def _legacy_calc_mpjpe_numpy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Replicates historical calc_mpjpe from sample.py."""
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=predicted.ndim - 1), axis=1)


def _legacy_loss_mpjpe_torch(predicted, target):
    """Replicates historical loss_mpjpe from sample.py."""
    import torch

    assert isinstance(predicted, torch.Tensor) and isinstance(target, torch.Tensor)
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=predicted.ndim - 1))


def test_compute_mpjpe_numpy_reductions():
    predicted = np.ones((2, 3, 4, 3), dtype=np.float32)
    target = np.zeros_like(predicted)

    per_joint = compute_mpjpe(predicted, target, reduce_axes="none")
    assert per_joint.shape == (2, 3, 4)
    assert per_joint.dtype == np.float32
    assert np.allclose(per_joint, np.sqrt(3), atol=1e-6)

    per_frame = compute_mpjpe(predicted, target, reduce_axes=("joint",))
    assert per_frame.shape == (2, 3)
    assert np.allclose(per_frame, np.sqrt(3), atol=1e-6)

    per_sequence = compute_mpjpe(predicted, target, reduce_axes=("joint", "time"))
    assert per_sequence.shape == (2,)
    assert np.allclose(per_sequence, np.sqrt(3), atol=1e-6)

    global_mean = compute_mpjpe(predicted, target, reduce_axes="global")
    assert np.isscalar(global_mean) or global_mean.shape == ()
    assert np.allclose(global_mean, np.sqrt(3), atol=1e-6)

    per_time = compute_mpjpe(predicted, target, reduce_axes=("batch", "joint"))
    assert per_time.shape == (3,)
    assert np.allclose(per_time, np.sqrt(3), atol=1e-6)


def test_compute_mpjpe_numpy_mask_and_weights():
    predicted = np.array([[[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]]], dtype=np.float32)
    target = np.zeros_like(predicted)
    mask = np.array([[[True, False]]])
    weights = np.array([[[1.0, 2.0]]], dtype=np.float32)

    masked = compute_mpjpe(predicted, target, joint_mask=mask, reduce_axes="global")
    assert np.allclose(masked, 3.0, atol=1e-6)

    weighted = compute_mpjpe(predicted, target, joint_weights=weights, reduce_axes="global")
    expected = (3.0 * 1.0 + 4.0 * 2.0) / (1.0 + 2.0)
    assert np.allclose(weighted, expected, atol=1e-6)

    masked_none = compute_mpjpe(predicted, target, joint_mask=mask, reduce_axes="none")
    assert masked_none.shape == (1, 1, 2)
    assert np.allclose(masked_none[..., 0], 3.0, atol=1e-6)
    assert np.all(masked_none[..., 1] == 0.0)


def test_compute_mpjpe_numpy_dtype_preserved():
    predicted = np.ones((1, 2, 2, 3), dtype=np.float32)
    target = np.zeros_like(predicted)
    result = compute_mpjpe(predicted, target, reduce_axes="none")
    assert result.dtype == np.float32

    predicted64 = predicted.astype(np.float64)
    target64 = target.astype(np.float64)
    result64 = compute_mpjpe(predicted64, target64, reduce_axes="global")
    assert result64.dtype == np.float64


def test_compute_mpjpe_shape_without_batch():
    predicted = np.ones((5, 2, 3), dtype=np.float32)
    target = np.zeros_like(predicted)
    per_joint = compute_mpjpe(predicted, target, reduce_axes="none")
    assert per_joint.shape == (5, 2)
    assert np.allclose(per_joint, np.sqrt(3), atol=1e-6)

    per_time = compute_mpjpe(predicted, target, reduce_axes=("joint",))
    assert per_time.shape == (5,)

    global_mean = compute_mpjpe(predicted, target, reduce_axes="global")
    assert np.allclose(global_mean, np.sqrt(3), atol=1e-6)


def test_compute_mpjpe_invalid_inputs():
    predicted = np.ones((1, 2, 2, 3), dtype=np.float32)
    target = np.ones((1, 3, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_mpjpe(predicted, target)

    bad = predicted.copy()
    bad[..., 0, 0] = np.nan
    with pytest.raises(ValueError):
        compute_mpjpe(bad, predicted)

    with pytest.raises(ValueError):
        compute_mpjpe(predicted[0, 0], predicted[0, 0])

    with pytest.raises(ValueError):
        compute_mpjpe(predicted[0], predicted[0], reduce_axes=("batch",))


def test_compute_mpjpe_matches_legacy_numpy():
    rng = np.random.default_rng(42)
    predicted = rng.standard_normal((2, 5, 6, 3), dtype=np.float32)
    target = rng.standard_normal((2, 5, 6, 3), dtype=np.float32)

    legacy = _legacy_calc_mpjpe_numpy(predicted, target)
    modern = compute_mpjpe(predicted, target, reduce_axes=("time",))
    assert modern.shape == legacy.shape == (2, 6)
    assert np.allclose(modern, legacy, atol=1e-6)


@pytest.mark.parametrize("reduction, expected_shape", [("mean", ()), ("none", (2,))])
def test_mpjpe_loss_torch(reduction, expected_shape):
    torch = pytest.importorskip("torch")

    predicted = torch.ones((2, 3, 4, 3), dtype=torch.float32)
    target = torch.zeros_like(predicted)

    loss_fn = MPJPELoss(reduction=reduction)
    result = loss_fn(predicted, target)
    assert result.shape == expected_shape
    if reduction == "none":
        expected = compute_mpjpe(predicted, target, reduce_axes=("time", "joint"))
        assert torch.allclose(result, expected)
    else:
        expected = compute_mpjpe(predicted, target, reduce_axes="global")
        assert torch.isclose(result, expected)


def test_mpjpe_loss_invalid_reduction():
    torch = pytest.importorskip("torch")
    predicted = torch.ones((2, 3, 4, 3), dtype=torch.float32)
    target = torch.zeros_like(predicted)
    with pytest.raises(ValueError):
        MPJPELoss(reduction="sum")(predicted, target)


def test_compute_mpjpe_torch_dtype_preserved():
    torch = pytest.importorskip("torch")
    predicted = torch.ones((1, 2, 3, 3), dtype=torch.float32)
    target = torch.zeros_like(predicted)
    result = compute_mpjpe(predicted, target, reduce_axes="none")
    assert result.dtype == torch.float32

    predicted_double = predicted.to(dtype=torch.float64)
    target_double = target.to(dtype=torch.float64)
    result_double = compute_mpjpe(predicted_double, target_double, reduce_axes="global")
    assert result_double.dtype == torch.float64


def test_mpjpe_loss_matches_legacy_torch():
    torch = pytest.importorskip("torch")

    predicted = torch.randn((3, 5, 6, 3), dtype=torch.float32)
    target = torch.randn((3, 5, 6, 3), dtype=torch.float32)

    legacy = _legacy_loss_mpjpe_torch(predicted, target)
    modern = mpjpe_loss(predicted, target, reduction="mean")
    assert torch.isclose(modern, legacy, atol=1e-6)


def test_torchmetrics_mpjpe_metric_matches_function():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")

    metric = MPJPE()
    predicted = torch.ones((2, 3, 4, 3), dtype=torch.float32)
    target = torch.zeros_like(predicted)

    metric.update(predicted, target)
    metric.update(predicted * 2, target)

    result = metric.compute()
    expected = compute_mpjpe(
        torch.cat([predicted, predicted * 2], dim=0), torch.zeros_like(torch.cat([predicted, predicted * 2], dim=0))
    )

    assert torch.isclose(result, expected)


def test_torchmetrics_mpjpe_rejects_invalid_inputs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")

    metric = MPJPE()
    bad = torch.ones((3, 4, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        metric.update(bad, bad)
