"""TorchMetrics Metric implementations for MPJPE and Procrustes-aligned MPJPE."""

from __future__ import annotations

try:
    from torchmetrics import Metric
except ImportError:  # pragma: no cover - torchmetrics optional
    Metric = None  # type: ignore[misc, assignment]

from ..metrics.mpjpe import compute_mpjpe, compute_p_mpjpe


def _require_torchmetrics() -> None:
    if Metric is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "torchmetrics is required to use pose3dkit.torchmetrics metrics."
        )


class MPJPE(Metric):  # type: ignore[misc]
    """TorchMetrics-compatible MPJPE returning a scalar global mean.

    Expects predictions and targets shaped (B, T, J, 3) and accumulates mean per-joint
    position error across updates. The metric averages across batch, time, and joint
    axes, keeping dtype/device compatibility with the inputs.
    """

    def __init__(self) -> None:
        _require_torchmetrics()
        # torchmetrics.Metric handles device placement of registered states.
        super().__init__(dist_sync_on_step=False)
        self._output_dtype = None
        # Keep accumulators in float64 for numerical stability; torchmetrics will cast
        # them to the current device automatically.
        self.add_state(
            "total_error",
            default=self._create_buffer(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=self._create_buffer(0.0),
            dist_reduce_fx="sum",
        )

    @staticmethod
    def _create_buffer(value: float):
        import torch

        return torch.tensor(float(value), dtype=torch.float64)

    def update(self, predicted, target) -> None:  # type: ignore[override]
        import torch

        if not isinstance(predicted, torch.Tensor) or not isinstance(
            target, torch.Tensor
        ):
            raise TypeError("MPJPE metric expects torch.Tensor inputs.")
        if predicted.shape != target.shape:
            raise ValueError(
                f"Predicted shape {tuple(predicted.shape)} must match target shape {tuple(target.shape)}."
            )
        if predicted.ndim != 4 or predicted.shape[-1] != 3:
            raise ValueError("Inputs must have shape (B, T, J, 3).")
        if not predicted.dtype.is_floating_point or not target.dtype.is_floating_point:
            raise TypeError("Inputs must use floating-point dtypes.")
        if not torch.isfinite(predicted).all():
            raise ValueError("Predicted tensor contains NaN or Inf values.")
        if not torch.isfinite(target).all():
            raise ValueError("Target tensor contains NaN or Inf values.")

        per_joint = compute_mpjpe(predicted, target, reduce_axes="none")
        self._output_dtype = per_joint.dtype
        # accumulate in float64 for numerical stability
        total = per_joint.sum(dtype=torch.float64)
        count = torch.tensor(
            per_joint.numel(), dtype=torch.float64, device=total.device
        )
        self.total_error += total.to(self.total_error.device)
        self.count += count.to(self.count.device)

    def compute(self):  # type: ignore[override]
        if self.count == 0:  # pragma: no cover - defensive
            raise ValueError("No samples have been provided; metric is undefined.")
        dtype = self._output_dtype or self.total_error.dtype  # type: ignore[attr-defined]
        return (self.total_error / self.count).to(dtype=dtype)


class PMPJPE(Metric):  # type: ignore[misc]
    """TorchMetrics-compatible PA-MPJPE returning a scalar global mean."""

    def __init__(self) -> None:
        _require_torchmetrics()
        super().__init__(dist_sync_on_step=False)
        self._output_dtype = None
        self.add_state(
            "total_error",
            default=self._create_buffer(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=self._create_buffer(0.0),
            dist_reduce_fx="sum",
        )

    @staticmethod
    def _create_buffer(value: float):
        import torch

        return torch.tensor(float(value), dtype=torch.float64)

    def update(self, predicted, target) -> None:  # type: ignore[override]
        import torch

        if not isinstance(predicted, torch.Tensor) or not isinstance(
            target, torch.Tensor
        ):
            raise TypeError("PMPJPE metric expects torch.Tensor inputs.")
        if predicted.shape != target.shape:
            raise ValueError(
                f"Predicted shape {tuple(predicted.shape)} must match target shape {tuple(target.shape)}."
            )
        if predicted.ndim != 4 or predicted.shape[-1] != 3:
            raise ValueError("Inputs must have shape (B, T, J, 3).")
        if not predicted.dtype.is_floating_point or not target.dtype.is_floating_point:
            raise TypeError("Inputs must use floating-point dtypes.")
        if not torch.isfinite(predicted).all():
            raise ValueError("Predicted tensor contains NaN or Inf values.")
        if not torch.isfinite(target).all():
            raise ValueError("Target tensor contains NaN or Inf values.")

        per_joint = compute_p_mpjpe(predicted, target, reduce_axes="none")
        self._output_dtype = per_joint.dtype
        total = per_joint.sum(dtype=torch.float64)
        count = torch.tensor(
            per_joint.numel(), dtype=torch.float64, device=total.device
        )
        self.total_error += total.to(self.total_error.device)
        self.count += count.to(self.count.device)

    def compute(self):  # type: ignore[override]
        if self.count == 0:  # pragma: no cover - defensive
            raise ValueError("No samples have been provided; metric is undefined.")
        dtype = self._output_dtype or self.total_error.dtype  # type: ignore[attr-defined]
        return (self.total_error / self.count).to(dtype=dtype)


__all__ = ["MPJPE", "PMPJPE"]
