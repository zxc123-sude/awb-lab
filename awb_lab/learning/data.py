"""Linear RGB preparation and explicitly grouped train/validation manifests."""

import csv
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..algorithms import _image, _unit_rgb
from ..evaluation import REQUIRED_COLUMNS
from ..image_io import load_mask, load_rgb

DATASET_TYPES = ("real", "synthetic_smoke_test")
PREPROCESSING = "linear_rgb_valid_weighted_area_resize_validmax_v1"


def canonical_path(path):
    return os.path.normcase(str(Path(path).resolve()))


def check_input_size(input_size):
    if type(input_size) is not int or not 16 <= input_size <= 1024:
        raise ValueError("input_size must be an integer between 16 and 1024")


def prepare_input(image, mask=None, input_size=128):
    """Mask BEFORE resize, use weighted interpolation, normalize valid exposure.

    Saturation (any normalized RGB channel >=1) is always invalid. Invalid
    values are set to zero before interpolation. RGB is divided by the single
    maximum across all valid RGB values, preserving channel ratios. Resize
    numerator and valid coverage separately, divide where coverage >0, and
    append soft coverage as channel four. Masked chart contents cannot leak.
    """
    check_input_size(input_size)
    array = _image(image)
    valid = np.max(array, axis=2) < 1.0
    if mask is not None:
        supplied = np.asarray(mask)
        if supplied.shape != array.shape[:2] or supplied.dtype.kind not in "bifu" or not np.isfinite(supplied).all():
            raise ValueError("mask must be finite H x W and match the RGB image")
        valid &= supplied != 0
    if not valid.any():
        raise ValueError("no valid unsaturated pixels remain")
    if (np.max(array[valid], axis=0) <= 0).any():
        raise ValueError("valid image is black or has a zero RGB channel")
    scale = float(np.max(array[valid]))
    safe = np.where(valid[..., None], array / scale, 0.0)
    coverage = cv2.resize(valid.astype(np.float64), (input_size, input_size), interpolation=cv2.INTER_AREA)
    numerator = cv2.resize(safe, (input_size, input_size), interpolation=cv2.INTER_AREA)
    rgb = np.divide(numerator, coverage[..., None], out=np.zeros_like(numerator), where=coverage[..., None] > 0)
    if not (coverage > 0).any():
        raise ValueError("no valid coverage remains after resizing")
    tensor = np.concatenate((rgb, coverage[..., None]), axis=2).transpose(2, 0, 1).astype(np.float32)
    if not np.isfinite(tensor).all():
        raise ValueError("nonfinite model input after preprocessing")
    return torch.from_numpy(np.ascontiguousarray(tensor))


def guard_synthetic(dataset_types, allow_synthetic):
    if "synthetic_smoke_test" in dataset_types and not allow_synthetic:
        raise ValueError("synthetic smoke-test data/checkpoint requires explicit --allow-synthetic; it is not a real-data AI benchmark")


def read_manifest(path, allow_synthetic=False):
    path = Path(path).resolve()
    required = set(REQUIRED_COLUMNS) | {"group_id", "dataset_type"}
    raw = path.read_bytes()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("learning manifest requires columns: " + ", ".join(sorted(required)))
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("learning manifest must contain image rows")
    records, image_paths, groups, dataset_types, grouping_units = [], set(), set(), set(), set()
    for line, row in enumerate(source_rows, 2):
        try:
            if any(not isinstance(row.get(column), str) or not row[column].strip() for column in required):
                raise ValueError("required values must be nonempty, including explicit black_level=0 if already corrected")
            image_path = canonical_path(path.parent / row["image"])
            if image_path in image_paths:
                raise ValueError("duplicate canonical image path in manifest")
            if row["dataset_type"] not in DATASET_TYPES:
                raise ValueError("dataset_type must be real or synthetic_smoke_test")
            black, white = float(row["black_level"]), float(row["white_level"])
            if not np.isfinite(black) or not np.isfinite(white) or white <= black:
                raise ValueError("levels must be finite with white_level > black_level")
            truth = _unit_rgb([float(row[f"illuminant_{c}"]) for c in "rgb"])
            group = row["group_id"].strip()
            grouping_unit = (row.get("grouping_unit") or "user_provided_group_ids_scene_semantics_unverified").strip()
            records.append({
                "image": image_path, "source_image": row["image"],
                "mask": canonical_path(path.parent / row["mask"]) if row.get("mask") else None,
                "ground_truth": truth, "black_level": black, "white_level": white,
                "group_id": group, "dataset_type": row["dataset_type"],
                "grouping_unit": grouping_unit,
                "sample_id": row.get("sample_id") or row["image"], "camera": row.get("camera") or "",
            })
            image_paths.add(image_path)
            groups.add(group)
            dataset_types.add(row["dataset_type"])
            grouping_units.add(grouping_unit)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"manifest {path.name} line {line}: {exc}") from exc
    guard_synthetic(dataset_types, allow_synthetic)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "records": records,
            "image_paths": sorted(image_paths), "groups": sorted(groups), "dataset_types": sorted(dataset_types),
            "grouping_units": sorted(grouping_units)}


def check_disjoint(first, second, first_name="train", second_name="validation"):
    path_overlap = set(first["image_paths"]) & set(second["image_paths"])
    group_overlap = set(first["groups"]) & set(second["groups"])
    if path_overlap or group_overlap:
        detail = "canonical image paths" if path_overlap else "group_id"
        raise ValueError(f"{first_name}/{second_name} overlap in {detail}; keep related captures/crops/cameras in the same group")


def load_record(record):
    image = load_rgb(Path(record["image"]), "linear", record["black_level"], record["white_level"])
    mask = load_mask(Path(record["mask"]), image.shape[:2]) if record["mask"] else None
    return image, mask


class IlluminantDataset(Dataset):
    def __init__(self, manifest, input_size=128):
        check_input_size(input_size)
        self.records, self.input_size = manifest["records"], input_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            image, mask = load_record(record)
            return prepare_input(image, mask, self.input_size), torch.tensor(record["ground_truth"], dtype=torch.float32)
        except (ValueError, OSError) as exc:
            raise ValueError(f"sample {record['sample_id']}: {exc}") from exc
