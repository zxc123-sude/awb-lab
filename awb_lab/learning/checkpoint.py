"""Versioned tensor-only loading plus a small, strictly validated schema."""

import hashlib
import math
import re
from pathlib import Path

import torch

from .data import DATASET_TYPES, PREPROCESSING, check_disjoint, check_input_size, guard_synthetic
from .model import ARCHITECTURE, TinyIlluminantNet

SCHEMA_VERSION = 1
META_KEYS = {
    "seed", "epoch", "val_mean_recovery_deg", "train_manifest_sha256", "val_manifest_sha256",
    "train_groups", "val_groups", "train_image_paths", "val_image_paths", "dataset_types",
    "synthetic", "torch_version", "preprocessing", "train_count", "val_count", "optimization",
    "train_grouping_units", "val_grouping_units",
}


def _validate_metadata(meta):
    if type(meta) is not dict or set(meta) != META_KEYS:
        raise ValueError("checkpoint metadata schema mismatch")
    for key in ("seed", "epoch", "train_count", "val_count"):
        if type(meta[key]) is not int or meta[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid checkpoint {key}")
    if type(meta["val_mean_recovery_deg"]) not in (int, float) or not math.isfinite(meta["val_mean_recovery_deg"]) or not 0 <= meta["val_mean_recovery_deg"] <= 180:
        raise ValueError("invalid checkpoint validation error")
    for key in ("train_manifest_sha256", "val_manifest_sha256"):
        if type(meta[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", meta[key]) is None:
            raise ValueError("invalid checkpoint manifest SHA256")
    for key in ("train_groups", "val_groups", "train_image_paths", "val_image_paths", "dataset_types", "train_grouping_units", "val_grouping_units"):
        values = meta[key]
        if type(values) is not list or not values or any(type(value) is not str or not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"invalid checkpoint {key}")
    if not set(meta["dataset_types"]).issubset(DATASET_TYPES):
        raise ValueError("invalid checkpoint dataset types")
    if type(meta["synthetic"]) is not bool or meta["synthetic"] != ("synthetic_smoke_test" in meta["dataset_types"]):
        raise ValueError("checkpoint synthetic provenance mismatch")
    if type(meta["torch_version"]) is not str or not meta["torch_version"] or meta["preprocessing"] != PREPROCESSING:
        raise ValueError("unsupported checkpoint preprocessing/version metadata")
    opt = meta["optimization"]
    if type(opt) is not dict or set(opt) != {"epochs", "batch_size", "learning_rate", "loss", "selection"}:
        raise ValueError("invalid checkpoint optimization metadata")
    if any(type(opt[key]) is not int or opt[key] < 1 for key in ("epochs", "batch_size")):
        raise ValueError("invalid checkpoint optimization integer")
    if type(opt["learning_rate"]) not in (int, float) or not math.isfinite(opt["learning_rate"]) or opt["learning_rate"] <= 0:
        raise ValueError("invalid checkpoint learning rate")
    if opt["loss"] != "one_minus_cosine" or opt["selection"] != "minimum_validation_mean_recovery_degrees":
        raise ValueError("unsupported checkpoint optimization protocol")
    if meta["epoch"] > opt["epochs"]:
        raise ValueError("checkpoint epoch exceeds configured training epochs")
    if len(meta["train_image_paths"]) != meta["train_count"] or len(meta["val_image_paths"]) != meta["val_count"]:
        raise ValueError("checkpoint sample count mismatch")
    check_disjoint({"groups": meta["train_groups"], "image_paths": meta["train_image_paths"]},
                   {"groups": meta["val_groups"], "image_paths": meta["val_image_paths"]})


def save_checkpoint(path, model, input_size, metadata):
    check_input_size(input_size)
    _validate_metadata(metadata)
    payload = {"schema_version": SCHEMA_VERSION, "architecture": ARCHITECTURE,
               "config": {"input_size": input_size}, "metadata": metadata,
               "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(path, allow_synthetic=False, device="cpu"):
    path = Path(path)
    # Never fall back to unrestricted pickle loading, even for old PyTorch.
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot safely load checkpoint with weights_only=True: {exc}") from exc
    if type(payload) is not dict or set(payload) != {"schema_version", "architecture", "config", "metadata", "state_dict"}:
        raise ValueError("checkpoint top-level schema mismatch")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION or payload["architecture"] != ARCHITECTURE:
        raise ValueError("unsupported checkpoint schema version or architecture")
    if type(payload["config"]) is not dict or set(payload["config"]) != {"input_size"}:
        raise ValueError("unsupported checkpoint configuration")
    check_input_size(payload["config"]["input_size"])
    _validate_metadata(payload["metadata"])
    guard_synthetic(payload["metadata"]["dataset_types"], allow_synthetic)
    state = payload["state_dict"]
    if type(state) is not dict or not state or any(type(key) is not str or not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for key, value in state.items()):
        raise ValueError("checkpoint state_dict must contain only finite named tensors")
    model = TinyIlluminantNet()
    reference = model.state_dict()
    if set(state) != set(reference) or any(state[key].shape != reference[key].shape or state[key].dtype != reference[key].dtype for key in reference):
        raise ValueError("checkpoint parameter shape/dtype schema mismatch")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint parameter schema mismatch: {exc}") from exc
    model.to(device).eval()
    return model, payload, hashlib.sha256(path.read_bytes()).hexdigest()
