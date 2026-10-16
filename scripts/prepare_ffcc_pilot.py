#!/usr/bin/env python3
"""Prepare an offline, fixed-fold real-data pilot from FFCC's tiny thumbnails.

Download the pinned FFCC source separately; this script never uses the network.
The author's thumbnails are already linear, black-corrected and chart-masked.
This pilot uses official fold 1 for train, 2 for validation and 3 for test;
official folds are NOT scene annotations or proof of scene independence.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


SOURCE_COMMIT = "2fa9e1316954dbd3913630b7d597927941b4dd32"
SOURCE_URL = "https://github.com/google/ffcc"
THUMB_SUBDIR = Path("data/shi_gehler/preprocessed/GehlerShiThumb")
GROUPING_UNIT = "official_fold_no_scene_annotations"
EXPECTED_FOLD_COUNTS = {1: 189, 2: 191, 3: 188}
SPLIT_NAMES = {1: "train", 2: "val", 3: "test"}
FIELDNAMES = (
    "image", "illuminant_r", "illuminant_g", "illuminant_b",
    "black_level", "white_level", "mask", "group_id", "grouping_unit",
    "dataset_type", "sample_id", "camera", "official_fold",
)


def _find_source(source_dir):
    source_dir = Path(source_dir).resolve(strict=True)
    if not source_dir.is_dir():
        raise ValueError("--source-dir must be a directory")
    candidates = [source_dir, source_dir / THUMB_SUBDIR]
    matches = [path for path in candidates if (path / "cvfolds.txt").is_file()]
    if len(matches) != 1:
        raise ValueError(
            "source must be the pinned FFCC checkout or its "
            "data/shi_gehler/preprocessed/GehlerShiThumb directory"
        )
    return matches[0]


def _checkout_commit(source_dir):
    """Verify a local checkout when available, without fetching anything."""
    root = next((p for p in (source_dir, *source_dir.parents)
                 if (p / ".git").exists()), None)
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot verify the source checkout's git revision") from exc
    revision = result.stdout.strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(
            f"source checkout revision is {revision}; expected {SOURCE_COMMIT}"
        )
    return revision


def _relative_path(path, base):
    try:
        return Path(os.path.relpath(str(path), str(base))).as_posix()
    except ValueError as exc:
        raise ValueError("source and output must support relative file paths") from exc


def prepare(source_dir, output_dir):
    """Validate all inputs before writing three manifests, masks and provenance."""
    source = _find_source(source_dir)
    output = Path(output_dir).resolve()
    source_revision = _checkout_commit(source)
    fold_path = source / "cvfolds.txt"
    fold_raw = fold_path.read_bytes()
    lines = fold_raw.decode("utf-8-sig").splitlines()
    if len(lines) != 568 or any(line.strip() not in {"1", "2", "3"} for line in lines):
        raise ValueError("cvfolds.txt must have exactly 568 lines containing 1, 2 or 3")
    folds = [int(line.strip()) for line in lines]
    fold_counts = {fold: folds.count(fold) for fold in SPLIT_NAMES}
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"unexpected official fold counts: {fold_counts}")
    expected_names = {f"{index:06d}.png" for index in range(1, 569)}
    actual_names = {path.name for path in source.glob("*.png")}
    if actual_names != expected_names:
        raise ValueError(
            "expected exactly 000001.png through 000568.png; "
            f"missing={sorted(expected_names - actual_names)[:5]}, "
            f"extra={sorted(actual_names - expected_names)[:5]}"
        )

    digest = hashlib.sha256()
    digest.update(b"cvfolds.txt\0" + fold_raw + b"\0")
    rows_by_split = {name: [] for name in SPLIT_NAMES.values()}
    masks = []
    zero_pixels = saturation_pixels = 0
    images_without_zero_pixels = []
    camera_counts = {}
    for index, fold in enumerate(folds, 1):
        sample = f"{index:06d}"
        image_path = source / f"{sample}.png"
        image_raw = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(image_raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (32, 48, 3) or image.dtype != np.uint8:
            raise ValueError(f"{image_path.name}: expected an 8-bit 32x48 three-channel PNG")
        # max across channels is identical for decoded BGR and RGB. The author's
        # masked chart/saturated pixels are zero; dark channels alone stay valid.
        maximum = np.max(image, axis=2)
        mask = (maximum > 0) & (maximum < 255)
        if not np.any(mask):
            raise ValueError(f"{image_path.name}: no valid nonzero unsaturated pixels")
        if (np.max(image[mask], axis=0) == 0).any():
            raise ValueError(f"{image_path.name}: a valid color channel is entirely zero")
        zero_count = int(np.count_nonzero(maximum == 0))
        zero_pixels += zero_count
        saturation_pixels += int(np.count_nonzero(maximum == 255))
        if zero_count == 0:
            images_without_zero_pixels.append(sample)

        gt_path = source / f"{sample}.txt"
        gt_raw = gt_path.read_bytes()
        try:
            values = gt_raw.decode("utf-8-sig").split()
            truth = np.asarray([float(value) for value in values], dtype=np.float64)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"{gt_path.name}: malformed RGB illuminant") from exc
        if truth.shape != (3,) or not np.isfinite(truth).all() or not np.all(truth > 0):
            raise ValueError(f"{gt_path.name}: expected exactly three finite positive RGB values")
        if not np.isclose(np.linalg.norm(truth), 1.0, atol=1e-6):
            raise ValueError(f"{gt_path.name}: expected the author's L2-normalized RGB illuminant")
        camera_path = source / f"{sample}_camera.txt"
        camera_raw = camera_path.read_bytes() if camera_path.is_file() else b""
        camera = camera_raw.decode("utf-8-sig").strip()
        if len(camera) > 200 or "\n" in camera or "\r" in camera:
            raise ValueError(f"{camera_path.name}: invalid camera metadata")
        camera_counts[camera or "not_provided"] = camera_counts.get(camera or "not_provided", 0) + 1
        for name, content in ((image_path.name, image_raw), (gt_path.name, gt_raw),
                              (camera_path.name, camera_raw)):
            digest.update(name.encode("utf-8") + b"\0" + content + b"\0")

        rows_by_split[SPLIT_NAMES[fold]].append({
            "image": _relative_path(image_path, output),
            **{f"illuminant_{channel}": format(float(value), ".17g")
               for channel, value in zip("rgb", truth)},
            "black_level": 0, "white_level": 255,
            "mask": f"masks/{sample}.npy", "group_id": f"official-fold-{fold}",
            "grouping_unit": GROUPING_UNIT, "dataset_type": "real",
            "sample_id": sample, "camera": camera, "official_fold": fold,
        })
        masks.append((sample, mask.astype(np.uint8)))

    provenance = {
        "dataset": "Gehler-Shi / FFCC GehlerShiThumb real-data pilot",
        "author_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_checkout_commit": source_revision,
        "source_revision_verified": source_revision == SOURCE_COMMIT,
        "source_directory": _relative_path(source, output),
        "source_files_sha256": digest.hexdigest(),
        "source_artifact": THUMB_SUBDIR.as_posix(),
        "sample_count": 568,
        "fold_counts": {str(fold): count for fold, count in fold_counts.items()},
        "split_protocol": {SPLIT_NAMES[fold]: f"official-fold-{fold}" for fold in SPLIT_NAMES},
        "grouping_unit": GROUPING_UNIT,
        "ground_truth": "Three positive RGB values from the author's corresponding .txt file, L2-normalized; illuminant, not gains.",
        "input": {
            "height": 32, "width": 48, "channels": 3, "dtype": "uint8",
            "color_space": "camera-linear RGB (no gamma, no camera CCM applied)",
            "black_level": 0, "white_level": 255,
            "author_preprocessing": "Subtract camera black level (Canon5D 129 / Canon1D 0), exclude color checker and saturation, mask-aware bilinear downsampling, per-image maximum normalization, and 8-bit quantization.",
            "additional_mask": "valid = (max(R,G,B)>0) & (max(R,G,B)<255); author-zeroed pixels and any channel at 255 are excluded. No black level is subtracted again.",
        },
        "mask_diagnostics": {
            "excluded_all_zero_pixels": zero_pixels,
            "excluded_any_255_pixels": saturation_pixels,
            "images_without_all_zero_pixels": images_without_zero_pixels,
            "note": "Masks are reconstructed from already masked author thumbnails; no original chart coordinates are recovered or inferred.",
        },
        "camera_counts": camera_counts,
        "limitations": [
            "This single train/validation/test pilot is not the paper's complete three-fold cross-validation reproduction.",
            "Official fold membership is not a scene annotation. Scene independence is unknown; group_id identifies only an official fold, not a physical scene.",
            "The mixed two-camera data does not establish generalization to unseen cameras.",
            "The 32x48 thumbnails have author downsampling, masking and 8-bit quantization; results do not establish full-resolution RAW or ISP deployment quality.",
            "The reconstructed mask trusts the pinned author's preprocessing and cannot independently verify every original color-chart boundary.",
        ],
        "references": [
            f"{SOURCE_URL}/blob/{SOURCE_COMMIT}/scripts/GehlerShiImportData.m",
            f"{SOURCE_URL}/blob/{SOURCE_COMMIT}/internal/ReadImage.m",
            f"{SOURCE_URL}/blob/{SOURCE_COMMIT}/projects/GehlerShiThumb/GehlerShiThumbConstants.m",
            "https://www.cs.sfu.ca/~colour/data/shi_gehler/",
        ],
        "attribution": "FFCC code: Apache-2.0. Cite Barron and Tsai, Fast Fourier Color Constancy (CVPR 2017), and the original Gehler / Shi-Funt dataset sources. Original dataset rights are not replaced by the code license.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    for sample, mask in masks:
        np.save(str(output / "masks" / f"{sample}.npy"), mask, allow_pickle=False)
    for name, rows in rows_by_split.items():
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    with (output / "provenance.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return {"output_dir": str(output), "samples": 568,
            "splits": {name: len(rows) for name, rows in rows_by_split.items()},
            "source_revision_verified": provenance["source_revision_verified"],
            "scene_independence": "unknown; official fold IDs only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="pinned FFCC checkout root or GehlerShiThumb directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.source_dir, args.output_dir)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(1, f"prepare_ffcc_pilot: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
