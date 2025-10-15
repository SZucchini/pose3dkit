from .metrics import (
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
]
