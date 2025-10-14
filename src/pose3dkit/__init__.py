from .metrics import MPJPELoss, compute_mpjpe, mpjpe_loss
from .torchmetrics import MPJPE as _TorchMetricsMPJPE

MPJPE = _TorchMetricsMPJPE

__all__ = ["compute_mpjpe", "mpjpe_loss", "MPJPELoss", "MPJPE"]
