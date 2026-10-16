"""Explicit transfer functions and bit-depth-preserving RGB image loading."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .algorithms import _image


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if (rgb < 0).any() or not np.isfinite(rgb).all():
        raise ValueError("linear RGB must be finite and nonnegative")
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(str(path), allow_pickle=False)
    # imdecode handles non-ASCII Windows filenames and retains uint16 RGB.
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise ValueError(f"cannot decode RGB image: {path}")
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("input must have exactly three RGB channels; remove alpha explicitly")
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def load_rgb(
    path: Path,
    input_space: str = "srgb",
    black_level: float = 0.0,
    white_level: Optional[float] = None,
) -> np.ndarray:
    """Load, subtract black, normalize by (white-black), optionally decode sRGB.

    Levels are in ORIGINAL file numeric units (e.g. 512/16383 for a 14-bit
    signal packed in uint16). Linear inputs always require explicit white
    level; dtype maximum is not assumed to be sensor saturation. Values
    below black and above white are clipped to [0,1] during preprocessing.
    NPY files must already be H x W x 3 RGB, never mosaiced Bayer RAW.
    """
    path = Path(path)
    if input_space not in ("srgb", "linear"):
        raise ValueError("input_space must be srgb or linear")
    if input_space == "linear" and path.suffix.lower() in (".jpg", ".jpeg"):
        raise ValueError("JPEG is a display demo only; use paired linear RGB PNG/TIFF/NPY for evaluation")
    if input_space == "linear" and white_level is None:
        raise ValueError("linear RGB requires an explicit white_level in original file units")
    raw = _read_image(path)
    array = _image(raw)
    if white_level is None:
        white_level = float(np.iinfo(raw.dtype).max) if raw.dtype.kind in "iu" else 1.0
    if not np.isfinite(black_level) or not np.isfinite(white_level) or white_level <= black_level:
        raise ValueError("levels must be finite, with white_level > black_level")
    normalized = np.clip((array - black_level) / (white_level - black_level), 0.0, 1.0)
    return srgb_to_linear(normalized) if input_space == "srgb" else normalized


def load_mask(path: Path, shape: tuple) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mask = np.load(str(path), allow_pickle=False)
    else:
        mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"cannot decode mask: {path}")
        if mask.ndim == 3:
            raise ValueError("mask must be a single-channel image; nonzero means valid")
    if mask.shape != shape or mask.dtype.kind not in "bifu" or not np.isfinite(mask).all():
        raise ValueError("mask must be finite H x W and match the RGB image")
    return mask != 0


def save_rgb(path: Path, linear_rgb: np.ndarray) -> None:
    path = Path(path)
    array = _image(linear_rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(str(path), array, allow_pickle=False)
        return
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("output must be .npy (linear) or .png/.jpg (8-bit sRGB preview)")
    preview = np.rint(np.clip(linear_to_srgb(np.clip(array, 0.0, 1.0)), 0.0, 1.0) * 255).astype(np.uint8)
    success, encoded = cv2.imencode(path.suffix, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError(f"cannot encode output {path}")
    encoded.tofile(str(path))
