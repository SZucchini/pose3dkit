"""Pose3D metric utilities.

This package exposes functions and loss classes related to Mean Per-Joint Position Error
(MPJPE) and related pose metrics used for 3D pose estimation, including procrustes-aligned
variants and velocity losses.
"""

from .mpjpe import (
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
]
