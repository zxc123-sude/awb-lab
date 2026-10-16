"""Statistics operate on linear RGB, before display gamma encoding.

These baselines assume a single global illuminant and a diagonal camera model.
They estimate an illuminant direction, not a calibrated color temperature.
"""

import math
from typing import Optional

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

METHODS = ("gray_world", "white_patch", "shades_of_gray", "gray_edge")


def _image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) == 0:
        raise ValueError("image must be a nonempty H x W x 3 RGB array")
    if array.dtype.kind not in "fiu":
        raise ValueError("image must contain real numeric RGB values")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("linear RGB must be finite and nonnegative")
    return array


def _positive_rgb(vector: np.ndarray, name: str = "illuminant") -> np.ndarray:
    rgb = np.asarray(vector, dtype=np.float64)
    if rgb.shape != (3,) or not np.isfinite(rgb).all() or (rgb <= 0).any():
        raise ValueError(f"{name} must have three finite, strictly positive RGB values")
    return rgb


def _unit_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = _positive_rgb(rgb)
    # Scale first to keep both very small and large finite values stable.
    scaled = rgb / np.max(rgb)
    return scaled / np.linalg.norm(scaled)


def _minkowski(pixels: np.ndarray, p: float) -> np.ndarray:
    scale = np.max(pixels, axis=0)
    if (scale <= 0).any():
        raise ValueError("selected pixels have no positive signal in at least one RGB channel")
    return scale * np.mean((pixels / scale) ** p, axis=0) ** (1.0 / p)


def estimate_illuminant(
    image: np.ndarray,
    method: str = "gray_world",
    mask: Optional[np.ndarray] = None,
    p: float = 6.0,
    sigma: float = 1.0,
    near_clip: Optional[float] = None,
) -> np.ndarray:
    """Return a positive L2-unit RGB illuminant estimated from linear RGB.

    ``mask`` is H x W: nonzero means valid. ``near_clip`` is an optional
    absolute intensity threshold in the SAME units as image; pixels with
    ANY channel >= threshold are excluded. It is not an image percentile.
    Gray Edge computes first-order Gaussian derivatives (truncate=4), then
    erodes validity by ceil(4*sigma) pixels to exclude masked/saturated
    neighborhoods and image borders. A flat image cannot support Gray Edge.
    """
    array = _image(image)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    if not np.isfinite(p) or p < 1:
        raise ValueError("p must be finite and >= 1")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0")
    valid = np.ones(array.shape[:2], dtype=bool)
    if mask is not None:
        raw_mask = np.asarray(mask)
        if raw_mask.shape != array.shape[:2] or raw_mask.dtype.kind not in "bifu":
            raise ValueError("mask must be a numeric/bool H x W array matching the image")
        if not np.isfinite(raw_mask).all():
            raise ValueError("mask must be finite")
        valid &= raw_mask != 0
    if near_clip is not None:
        if not np.isfinite(near_clip) or near_clip <= 0:
            raise ValueError("near_clip must be finite and > 0")
        valid &= np.max(array, axis=2) < near_clip
    if not valid.any():
        raise ValueError("no valid pixels remain after mask and clipping exclusion")

    if method == "gray_edge":
        radius = math.ceil(4.0 * sigma)
        if min(array.shape[:2]) <= 2 * radius:
            raise ValueError("image is too small for Gray Edge derivative support")
        # Gaussian derivatives have a finite square support. Eroding the
        # original mask prevents excluded chart edges entering the estimate.
        valid = binary_erosion(valid, structure=np.ones((3, 3), bool), iterations=radius)
        if not valid.any():
            raise ValueError("no valid Gray Edge neighborhoods remain; use a larger image or smaller sigma")
        dx = gaussian_filter(array, sigma=(sigma, sigma, 0), order=(0, 1, 0), truncate=4.0)
        dy = gaussian_filter(array, sigma=(sigma, sigma, 0), order=(1, 0, 0), truncate=4.0)
        statistics = _minkowski(np.hypot(dx, dy)[valid], p)
    elif method == "white_patch":
        statistics = np.max(array[valid], axis=0)
    elif method == "gray_world":
        statistics = _minkowski(array[valid], 1.0)
    else:
        statistics = _minkowski(array[valid], p)
    return _unit_rgb(statistics)


def apply_white_balance(image: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    """Return linear RGB with green-normalized diagonal gains e_G/e_c.

    No output clipping, gamma conversion, or global exposure normalization is
    performed. Values may exceed the input white level after correction.
    """
    array = _image(image)
    rgb = _positive_rgb(illuminant)
    gains = rgb[1] / rgb
    result = array * gains
    if not np.isfinite(result).all():
        raise ValueError("white balance exceeds floating-point range")
    return result
