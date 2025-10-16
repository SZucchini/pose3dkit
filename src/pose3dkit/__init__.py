"""Top-level exports for pose3dkit metrics and TorchMetrics integrations."""

from .metrics import (
    MPJPELoss,
    NMPJPELoss,
    PMPJPELoss,
    VelocityLoss,
    compute_mpjpe,
    compute_p_mpjpe,
    mpjpe_loss,
    n_mpjpe_loss,
    p_mpjpe_loss,
    velocity_loss,
)
from .normalization import (
    denormalize_keypoints,
    denormalize_keypoints_2d,
    denormalize_keypoints_3d,
    normalize_keypoints,
    normalize_keypoints_2d,
    normalize_keypoints_3d,
)
from .torchmetrics import MPJPE as _TorchMetricsMPJPE
from .torchmetrics import PMPJPE as _TorchMetricsPMPJPE

MPJPE = _TorchMetricsMPJPE
PMPJPE = _TorchMetricsPMPJPE

__all__ = [
    "compute_mpjpe",
    "compute_p_mpjpe",
    "mpjpe_loss",
    "p_mpjpe_loss",
    "n_mpjpe_loss",
    "velocity_loss",
    "MPJPELoss",
    "PMPJPELoss",
    "NMPJPELoss",
    "VelocityLoss",
    "MPJPE",
    "PMPJPE",
    "normalize_keypoints",
    "denormalize_keypoints",
    "normalize_keypoints_2d",
    "denormalize_keypoints_2d",
    "normalize_keypoints_3d",
    "denormalize_keypoints_3d",
]
