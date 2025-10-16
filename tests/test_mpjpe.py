import numpy as np
import pytest

from pose3dkit import MPJPE, PMPJPE
from pose3dkit.metrics import (
    MPJPELoss,
    NMPJPELoss,
    PMPJPELoss,
    VelocityLoss,
    compute_mpjpe,
    compute_p_mpjpe,
    mpjpe_loss,
    p_mpjpe_loss,
    n_mpjpe_loss,
    velocity_loss,
)


def _legacy_calc_mpjpe_numpy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Replicates historical calc_mpjpe."""
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=predicted.ndim - 1), axis=1)


def _legacy_loss_mpjpe_torch(predicted, target):
    """Replicates historical loss_mpjpe."""
    import torch

    assert isinstance(predicted, torch.Tensor) and isinstance(target, torch.Tensor)
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=predicted.ndim - 1))


def _legacy_p_mpjpe_numpy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Replicates historical p_mpjpe with optional batching."""

    def _single(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        mu_x = np.mean(tgt, axis=1, keepdims=True)
        mu_y = np.mean(pred, axis=1, keepdims=True)

        x0 = tgt - mu_x
        y0 = pred - mu_y

        norm_x = np.sqrt(np.sum(x0**2, axis=(1, 2), keepdims=True))
        norm_y = np.sqrt(np.sum(y0**2, axis=(1, 2), keepdims=True))
        x0 /= norm_x
        y0 /= norm_y

        h = np.matmul(np.transpose(x0, (0, 2, 1)), y0)
        u, s, v_t = np.linalg.svd(h, full_matrices=False)
        v = np.transpose(v_t, (0, 2, 1))
        r = np.matmul(v, np.transpose(u, (0, 2, 1)))

        sign_det = np.sign(np.linalg.det(r))
        sign_det[sign_det == 0] = 1.0
        v[:, :, -1] *= sign_det[:, np.newaxis]
        s[:, -1] *= sign_det
        r = np.matmul(v, np.transpose(u, (0, 2, 1)))

        tr = np.sum(s, axis=1, keepdims=True)[:, :, np.newaxis]
        a = tr * norm_x / norm_y
        t = mu_x - a * np.matmul(mu_y, r)
        aligned = a * np.matmul(pred, r) + t
        return np.mean(np.linalg.norm(aligned - tgt, axis=2), axis=1)

    if predicted.ndim == 3:
        return _single(predicted, target)
    return np.stack(
        [_single(predicted[b], target[b]) for b in range(predicted.shape[0])], axis=0
    )


def _legacy_p_mpjpe_torch(predicted, target):
    """Replicates historical p_mpjpe with optional batching."""
    import torch

    assert isinstance(predicted, torch.Tensor) and isinstance(target, torch.Tensor)
    assert predicted.shape == target.shape

    add_batch = False
    if predicted.ndim == 3:
        predicted = predicted.unsqueeze(0)
        target = target.unsqueeze(0)
        add_batch = True

    pred64 = predicted.to(torch.float64)
    tgt64 = target.to(torch.float64)

    mu_x = tgt64.mean(dim=2, keepdim=True)
    mu_y = pred64.mean(dim=2, keepdim=True)

    x0 = tgt64 - mu_x
    y0 = pred64 - mu_y

    norm_x = torch.norm(x0, dim=(2, 3), keepdim=True)
    norm_y = torch.norm(y0, dim=(2, 3), keepdim=True)
    x0 = x0 / norm_x
    y0 = y0 / norm_y

    h = torch.matmul(x0.transpose(-2, -1), y0)
    u, s, v_t = torch.linalg.svd(h, full_matrices=False)
    v = v_t.transpose(-2, -1)
    r = torch.matmul(v, u.transpose(-2, -1))

    sign_det_r = torch.sign(torch.linalg.det(r)).unsqueeze(-1)
    sign_det_r[sign_det_r == 0] = 1.0
    v[..., :, -1] *= sign_det_r
    s[..., -1] *= sign_det_r.squeeze(-1)
    r = torch.matmul(v, u.transpose(-2, -1))

    tr = s.sum(dim=-1, keepdim=True)
    a = tr.unsqueeze(-1) * norm_x / norm_y
    t = mu_x - a * torch.matmul(mu_y, r)

    aligned = a * torch.matmul(pred64, r) + t
    errors = torch.norm(aligned - tgt64, dim=3).mean(dim=2)
    errors = errors.to(predicted.dtype)

    if add_batch:
        return errors.squeeze(0)
    return errors


def _legacy_n_mpjpe_torch(predicted, target):
    """Replicates historical n_mpjpe."""
    import torch

    assert isinstance(predicted, torch.Tensor) and isinstance(target, torch.Tensor)
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(
        torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True
    )
    norm_target = torch.mean(
        torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True
    )
    scale = norm_target / norm_predicted
    return torch.mean(torch.norm(scale * predicted - target, dim=3))


def _legacy_velocity_loss(predicted, target):
    """Replicates historical loss_velocity."""
    import torch

    assert isinstance(predicted, torch.Tensor) and isinstance(target, torch.Tensor)
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.0)[0].to(predicted.device)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))


def test_compute_mpjpe_numpy_reductions():
    np.random.default_rng(0)
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

    weighted = compute_mpjpe(
        predicted, target, joint_weights=weights, reduce_axes="global"
    )
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


def test_compute_p_mpjpe_numpy_matches_legacy():
    rng = np.random.default_rng(7)
    predicted = rng.standard_normal((2, 4, 5, 3), dtype=np.float32)
    target = rng.standard_normal((2, 4, 5, 3), dtype=np.float32)

    legacy = _legacy_p_mpjpe_numpy(predicted, target)
    modern = compute_p_mpjpe(predicted, target, reduce_axes=("joint",))
    assert modern.shape == legacy.shape
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


def test_compute_mpjpe_torch_bfloat16_support():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((2, 3, 4, 3), dtype=torch.bfloat16)
    target = torch.randn((2, 3, 4, 3), dtype=torch.bfloat16)

    result_none = compute_mpjpe(predicted, target, reduce_axes="none")
    assert result_none.dtype == torch.bfloat16

    result_global = compute_mpjpe(predicted, target, reduce_axes="global")
    assert isinstance(result_global, torch.Tensor)
    assert result_global.dtype == torch.bfloat16

    reference = compute_mpjpe(
        predicted.to(torch.float32), target.to(torch.float32), reduce_axes="global"
    )
    assert torch.isclose(result_global.to(torch.float32), reference, atol=2e-2)


def test_compute_p_mpjpe_torch_matches_legacy():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((2, 4, 5, 3), dtype=torch.float32)
    target = torch.randn((2, 4, 5, 3), dtype=torch.float32)

    legacy = _legacy_p_mpjpe_torch(predicted, target)
    modern = compute_p_mpjpe(predicted, target, reduce_axes=("joint",))
    assert torch.allclose(modern, legacy, atol=1e-6)


def test_mpjpe_loss_matches_legacy_torch():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((3, 5, 6, 3), dtype=torch.float32)
    target = torch.randn((3, 5, 6, 3), dtype=torch.float32)

    legacy = _legacy_loss_mpjpe_torch(predicted, target)
    modern = mpjpe_loss(predicted, target, reduction="mean")
    assert torch.isclose(modern, legacy, atol=1e-6)


def test_mpjpe_loss_supports_bfloat16():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((2, 4, 5, 3), dtype=torch.bfloat16)
    target = torch.randn((2, 4, 5, 3), dtype=torch.bfloat16)

    mean_loss = mpjpe_loss(predicted, target, reduction="mean")
    assert mean_loss.dtype == torch.bfloat16

    reference_mean = mpjpe_loss(
        predicted.to(torch.float32), target.to(torch.float32), reduction="mean"
    )
    assert torch.isclose(mean_loss.to(torch.float32), reference_mean, atol=2e-2)

    none_loss = mpjpe_loss(predicted, target, reduction="none")
    assert none_loss.dtype == torch.bfloat16
    assert none_loss.shape == (predicted.shape[0],)

    reference_none = mpjpe_loss(
        predicted.to(torch.float32), target.to(torch.float32), reduction="none"
    )
    assert torch.allclose(none_loss.to(torch.float32), reference_none, atol=2e-2)


def test_p_mpjpe_loss_matches_legacy():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((3, 5, 6, 3), dtype=torch.float32)
    target = torch.randn((3, 5, 6, 3), dtype=torch.float32)

    legacy = _legacy_p_mpjpe_torch(predicted, target)
    modern_mean = p_mpjpe_loss(predicted, target, reduction="mean")
    modern_none = p_mpjpe_loss(predicted, target, reduction="none")

    assert torch.isclose(modern_mean, legacy.mean(), atol=1e-6)
    assert torch.allclose(modern_none, legacy.mean(dim=1), atol=1e-6)


def test_p_mpjpe_loss_supports_bfloat16():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((2, 4, 5, 3), dtype=torch.bfloat16)
    target = torch.randn((2, 4, 5, 3), dtype=torch.bfloat16)

    mean_loss = p_mpjpe_loss(predicted, target, reduction="mean")
    none_loss = p_mpjpe_loss(predicted, target, reduction="none")

    assert mean_loss.dtype == torch.bfloat16
    assert none_loss.dtype == torch.bfloat16
    assert none_loss.shape == (predicted.shape[0],)

    ref_mean = p_mpjpe_loss(
        predicted.to(torch.float32), target.to(torch.float32), reduction="mean"
    )
    ref_none = p_mpjpe_loss(
        predicted.to(torch.float32), target.to(torch.float32), reduction="none"
    )

    assert torch.isclose(mean_loss.to(torch.float32), ref_mean, atol=2e-2)
    assert torch.allclose(none_loss.to(torch.float32), ref_none, atol=2e-2)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_n_mpjpe_loss_matches_legacy(dtype):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    predicted = torch.randn((2, 4, 5, 3), dtype=torch_dtype)
    target = torch.randn((2, 4, 5, 3), dtype=torch_dtype)

    legacy = _legacy_n_mpjpe_torch(predicted, target)
    modern = n_mpjpe_loss(predicted, target, reduction="mean")
    assert torch.isclose(modern.to(torch.float32), legacy.to(torch.float32), atol=1e-2)


def test_n_mpjpe_loss_none_matches_manual():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((3, 4, 6, 3), dtype=torch.float32)
    target = torch.randn((3, 4, 6, 3), dtype=torch.float32)

    result = n_mpjpe_loss(predicted, target, reduction="none")

    norm_predicted = torch.mean(
        torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True
    )
    norm_target = torch.mean(
        torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True
    )
    scale = norm_target / norm_predicted
    manual = torch.norm(scale * predicted - target, dim=3).mean(dim=(1, 2))

    assert torch.allclose(result, manual, atol=1e-6)


def test_nmpjpe_loss_module_matches_function():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((2, 3, 4, 3), dtype=torch.float32)
    target = torch.randn((2, 3, 4, 3), dtype=torch.float32)

    module = NMPJPELoss(reduction="none")
    assert torch.allclose(
        module(predicted, target),
        n_mpjpe_loss(predicted, target, reduction="none"),
        atol=1e-6,
    )


def test_pmpjpe_loss_module_matches_function():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((2, 3, 4, 3), dtype=torch.float32)
    target = torch.randn((2, 3, 4, 3), dtype=torch.float32)

    module = PMPJPELoss(reduction="none")
    assert torch.allclose(
        module(predicted, target),
        p_mpjpe_loss(predicted, target, reduction="none"),
        atol=1e-6,
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_velocity_loss_matches_legacy(dtype):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    predicted = torch.randn((3, 5, 6, 3), dtype=torch_dtype)
    target = torch.randn((3, 5, 6, 3), dtype=torch_dtype)

    legacy = _legacy_velocity_loss(predicted, target)
    modern = velocity_loss(predicted, target, reduction="mean")
    assert torch.isclose(modern.to(torch.float32), legacy.to(torch.float32), atol=1e-2)


def test_velocity_loss_none_reduction():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((4, 6, 7, 3), dtype=torch.float32)
    target = torch.randn((4, 6, 7, 3), dtype=torch.float32)

    result = velocity_loss(predicted, target, reduction="none")
    assert result.shape == (4,)

    legacy = _legacy_velocity_loss(predicted, target)
    assert torch.isclose(result.mean(), legacy, atol=1e-6)


def test_velocity_loss_short_sequence_returns_zero():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((2, 1, 5, 3), dtype=torch.float32)
    target = torch.randn((2, 1, 5, 3), dtype=torch.float32)

    assert torch.allclose(
        velocity_loss(predicted, target, reduction="none"),
        torch.zeros(2, dtype=torch.float32),
    )
    assert torch.isclose(
        velocity_loss(predicted, target, reduction="mean"), torch.tensor(0.0)
    )


def test_velocity_loss_module_matches_function():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    predicted = torch.randn((3, 5, 6, 3), dtype=torch.float32)
    target = torch.randn((3, 5, 6, 3), dtype=torch.float32)

    module = VelocityLoss(reduction="none")
    assert torch.allclose(
        module(predicted, target),
        velocity_loss(predicted, target, reduction="none"),
        atol=1e-6,
    )


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
        torch.cat([predicted, predicted * 2], dim=0),
        torch.zeros_like(torch.cat([predicted, predicted * 2], dim=0)),
    )

    assert torch.isclose(result, expected)


def test_torchmetrics_mpjpe_rejects_invalid_inputs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")

    metric = MPJPE()
    bad = torch.ones((3, 4, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        metric.update(bad, bad)


def test_torchmetrics_pmpjpe_metric_matches_function():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")

    metric = PMPJPE()
    torch.manual_seed(0)

    predicted = torch.randn((2, 3, 4, 3), dtype=torch.float32)
    target = torch.randn((2, 3, 4, 3), dtype=torch.float32)

    metric.update(predicted, target)
    metric.update(predicted * 1.5, target)

    stacked_pred = torch.cat([predicted, predicted * 1.5], dim=0)
    stacked_target = torch.cat([target, target], dim=0)
    result = metric.compute()
    expected = compute_p_mpjpe(stacked_pred, stacked_target, reduce_axes="global")

    assert torch.isclose(result, expected)
