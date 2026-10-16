"""Angular errors, in degrees; ground truth is paired camera-RGB illuminant."""

import numpy as np

from .algorithms import _positive_rgb, _unit_rgb


def recovery_angular_error(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    """Angle between estimated and ground-truth illuminant directions."""
    cosine = float(np.dot(_unit_rgb(estimate), _unit_rgb(ground_truth)))
    # acos's derivative diverges at 1; explicitly preserve exact identity.
    if cosine >= 1.0 - 4 * np.finfo(float).eps:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def reproduction_angular_error(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    """Angle between corrected ground-truth white (gt/estimate) and [1,1,1]."""
    est = _unit_rgb(_positive_rgb(estimate, "estimate"))
    gt = _unit_rgb(_positive_rgb(ground_truth, "ground_truth"))
    return recovery_angular_error(gt / est, np.ones(3))
