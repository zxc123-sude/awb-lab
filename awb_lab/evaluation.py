"""Paired linear RGB evaluation with explicit provenance and numeric levels."""

import csv
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

import numpy as np

from .algorithms import METHODS, estimate_illuminant
from .image_io import load_mask, load_rgb
from .metrics import recovery_angular_error, reproduction_angular_error

REQUIRED_COLUMNS = ("image", "illuminant_r", "illuminant_g", "illuminant_b", "black_level", "white_level")


def _stats(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=float)
    worst_count = max(1, math.ceil(0.25 * len(array)))
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "worst_25_percent_mean": float(np.mean(np.sort(array)[-worst_count:])),
    }


def evaluate(
    manifest: Path,
    output_dir: Path,
    methods: Sequence[str] = METHODS,
    p: float = 6.0,
    sigma: float = 1.0,
    near_clip: Optional[float] = 1.0,
) -> dict:
    """Evaluate every manifest row; fail on invalid rows instead of dropping them.

    image/mask paths are relative to the CSV. mask is optional, nonzero=valid.
    black_level and white_level are REQUIRED in raw file units. Already
    black-corrected inputs must explicitly specify black_level=0.
    Ground-truth RGB MUST already correspond to black-corrected image data;
    no black subtraction or color-space transform is applied to illuminants.
    Optional sample_id/camera/split/dataset_type are copied to predictions.
    Parameters are fixed across rows; this function does not fit or tune them.
    Evaluation always excludes saturation: None resolves to 1.0 and an
    explicit threshold must be in (0,1] after black/white normalization.
    """
    manifest = Path(manifest).resolve()
    output_dir = Path(output_dir)
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a nonempty unique subset of {METHODS}")
    effective_near_clip = 1.0 if near_clip is None else near_clip
    if not np.isfinite(effective_near_clip) or not 0 < effective_near_clip <= 1:
        raise ValueError("evaluation near_clip must be finite and in (0, 1]")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(REQUIRED_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"manifest requires columns: {', '.join(REQUIRED_COLUMNS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest has no image rows")

    predictions = []
    for line_number, row in enumerate(rows, 2):
        try:
            if not row["image"] or not row["black_level"] or not row["white_level"]:
                raise ValueError("image, black_level and white_level must be nonempty")
            image_path = manifest.parent / row["image"]
            image = load_rgb(
                image_path, input_space="linear",
                black_level=float(row["black_level"]),
                white_level=float(row["white_level"]),
            )
            mask = load_mask(manifest.parent / row["mask"], image.shape[:2]) if row.get("mask") else None
            ground_truth = np.array([float(row[f"illuminant_{channel}"]) for channel in "rgb"])
            # Validate paired ground truth before spending time on algorithms.
            recovery_angular_error(ground_truth, ground_truth)
            for method in methods:
                start = perf_counter()
                estimate = estimate_illuminant(image, method, mask, p, sigma, effective_near_clip)
                elapsed_ms = (perf_counter() - start) * 1000
                predictions.append({
                    "sample_id": row.get("sample_id") or row["image"],
                    "image": row["image"], "camera": row.get("camera", ""),
                    "split": row.get("split", ""), "dataset_type": row.get("dataset_type", ""),
                    "method": method,
                    "estimate_r": float(estimate[0]), "estimate_g": float(estimate[1]),
                    "estimate_b": float(estimate[2]),
                    "recovery_error_deg": recovery_angular_error(estimate, ground_truth),
                    "reproduction_error_deg": reproduction_angular_error(estimate, ground_truth),
                    "runtime_ms": elapsed_ms,
                })
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise ValueError(f"manifest line {line_number} ({row.get('image', '')}): {exc}") from exc

    summary = {
        "manifest": str(manifest), "sample_count": len(rows), "failed_count": 0,
        "dataset_types": sorted({row.get("dataset_type") or "unspecified" for row in rows}),
        "input_space": "linear", "gt_convention": "paired black-level-corrected camera RGB",
        "mask_convention": "nonzero=valid, zero=excluded",
        "parameters": {"p": p, "sigma": sigma, "near_clip": effective_near_clip},
        "timing_note": "Single run per image; decode excluded, mask/validation included. Not a controlled performance benchmark.",
        "methods": {},
    }
    for method in methods:
        selected = [row for row in predictions if row["method"] == method]
        summary["methods"][method] = {
            "count": len(selected),
            "recovery_error_deg": _stats([row["recovery_error_deg"] for row in selected]),
            "reproduction_error_deg": _stats([row["reproduction_error_deg"] for row in selected]),
            "runtime_ms": _stats([row["runtime_ms"] for row in selected]),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with (output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return summary
