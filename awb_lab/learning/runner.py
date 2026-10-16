"""Train on train only; select on validation only; compare on held-out groups."""

import csv
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..algorithms import METHODS, apply_white_balance, estimate_illuminant
from ..evaluation import _stats
from ..image_io import load_mask, load_rgb, save_rgb
from ..metrics import recovery_angular_error, reproduction_angular_error
from .checkpoint import load_checkpoint, save_checkpoint
from .data import (PREPROCESSING, IlluminantDataset, canonical_path, check_disjoint,
                   check_input_size, load_record, prepare_input, read_manifest)
from .model import TinyIlluminantNet, cosine_loss


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _device(value):
    device = torch.device(value)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("device must be cpu or an available CUDA device")
    return device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validation(model, loader, device):
    model.eval()
    errors, weighted_loss, count = [], 0.0, 0
    with torch.inference_mode():
        for images, truth in loader:
            images, truth = images.to(device), truth.to(device)
            estimate = model(images)
            loss = cosine_loss(estimate, truth)
            weighted_loss += float(loss.item()) * len(images)
            count += len(images)
            for pred, target in zip(estimate.cpu().numpy(), truth.cpu().numpy()):
                errors.append(recovery_angular_error(pred, target))
    return weighted_loss / count, float(np.mean(errors))


def train(train_csv, val_csv, output_dir, epochs=20, batch_size=8, input_size=128,
          seed=42, device="cpu", learning_rate=1e-3, allow_synthetic=False):
    check_input_size(input_size)
    if any(type(value) is not int or value < 1 for value in (epochs, batch_size)):
        raise ValueError("epochs and batch_size must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and > 0")
    train_manifest = read_manifest(train_csv, allow_synthetic)
    val_manifest = read_manifest(val_csv, allow_synthetic)
    check_disjoint(train_manifest, val_manifest)
    device = _device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(seed)
    training = DataLoader(IlluminantDataset(train_manifest, input_size), batch_size=batch_size,
                          shuffle=True, generator=generator, num_workers=0)
    validation = DataLoader(IlluminantDataset(val_manifest, input_size), batch_size=batch_size,
                            shuffle=False, num_workers=0)
    model = TinyIlluminantNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    initial_val_loss, initial_val_error = _validation(model, validation, device)
    best_error, best_epoch, history = math.inf, 0, []
    dataset_types = sorted(set(train_manifest["dataset_types"]) | set(val_manifest["dataset_types"]))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for images, truth in training:
            images, truth = images.to(device), truth.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = cosine_loss(model(images), truth)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise ValueError("nonfinite training gradient")
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            count += len(images)
        val_loss, val_error = _validation(model, validation, device)
        history.append({"epoch": epoch, "train_cosine_loss": total_loss / count,
                        "val_cosine_loss": val_loss, "val_mean_recovery_deg": val_error})
        print(f"epoch {epoch}/{epochs}: train_loss={total_loss / count:.6f}, val_mean_angle={val_error:.4f} deg", file=sys.stderr, flush=True)
        if val_error < best_error:
            best_error, best_epoch = val_error, epoch
            metadata = {
                "seed": seed, "epoch": epoch, "val_mean_recovery_deg": val_error,
                "train_manifest_sha256": train_manifest["sha256"], "val_manifest_sha256": val_manifest["sha256"],
                "train_groups": train_manifest["groups"], "val_groups": val_manifest["groups"],
                "train_grouping_units": train_manifest["grouping_units"], "val_grouping_units": val_manifest["grouping_units"],
                "train_image_paths": train_manifest["image_paths"], "val_image_paths": val_manifest["image_paths"],
                "dataset_types": dataset_types, "synthetic": "synthetic_smoke_test" in dataset_types,
                "torch_version": str(torch.__version__), "preprocessing": PREPROCESSING,
                "train_count": len(train_manifest["records"]), "val_count": len(val_manifest["records"]),
                "optimization": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                                 "loss": "one_minus_cosine", "selection": "minimum_validation_mean_recovery_degrees"},
            }
            save_checkpoint(checkpoint, model, input_size, metadata)
    report = {"checkpoint": str(checkpoint), "best_epoch": best_epoch,
              "best_val_mean_recovery_deg": best_error, "initial_val_mean_recovery_deg": initial_val_error,
              "initial_val_cosine_loss": initial_val_loss, "history": history,
              "dataset_types": dataset_types, "synthetic": "synthetic_smoke_test" in dataset_types,
              "train_grouping_units": train_manifest["grouping_units"], "val_grouping_units": val_manifest["grouping_units"],
              "interpretation": "Synthetic smoke test only; no real-data accuracy claim" if "synthetic_smoke_test" in dataset_types else "Validation-selected model; held-out testing still required"}
    _json(output_dir / "training.json", report)
    return report


def evaluate(checkpoint, test_csv, output_dir, device="cpu", allow_synthetic=False):
    device = _device(device)
    model, payload, checkpoint_hash = load_checkpoint(checkpoint, allow_synthetic, device)
    manifest = read_manifest(test_csv, allow_synthetic)
    meta, input_size = payload["metadata"], payload["config"]["input_size"]
    for split in ("train", "val"):
        seen = {"groups": meta[f"{split}_groups"], "image_paths": meta[f"{split}_image_paths"]}
        check_disjoint(seen, manifest, split, "heldout_test")
    predictions = []
    for record in manifest["records"]:
        try:
            image, mask = load_record(record)
            start = perf_counter()
            tensor = prepare_input(image, mask, input_size).unsqueeze(0).to(device)
            _sync(device)
            prepared_ms = (perf_counter() - start) * 1000
            _sync(device)
            start = perf_counter()
            with torch.inference_mode():
                estimate = model(tensor)
            _sync(device)
            forward_ms = (perf_counter() - start) * 1000
            estimates = [("tiny_cnn", estimate[0].cpu().numpy(), prepared_ms + forward_ms,
                          f"{input_size}x{input_size}", prepared_ms, forward_ms)]
            for method in METHODS:
                start = perf_counter()
                estimate = estimate_illuminant(image, method, mask=mask, p=6, sigma=1, near_clip=1.0)
                estimates.append((method, estimate, (perf_counter() - start) * 1000,
                                  f"{image.shape[0]}x{image.shape[1]}", "", ""))
            for method, estimate, elapsed, size, prep, forward in estimates:
                predictions.append({"sample_id": record["sample_id"], "image": record["source_image"],
                                    "group_id": record["group_id"], "camera": record["camera"],
                                    "dataset_type": record["dataset_type"], "method": method,
                                    "estimate_r": float(estimate[0]), "estimate_g": float(estimate[1]), "estimate_b": float(estimate[2]),
                                    "recovery_error_deg": recovery_angular_error(estimate, record["ground_truth"]),
                                    "reproduction_error_deg": reproduction_angular_error(estimate, record["ground_truth"]),
                                    "algorithm_input_size": size, "runtime_ms": elapsed,
                                    "cnn_preprocess_transfer_ms": prep, "cnn_forward_ms": forward})
        except (ValueError, OSError, RuntimeError) as exc:
            raise ValueError(f"heldout sample {record['sample_id']}: {exc}") from exc
    synthetic = meta["synthetic"] or "synthetic_smoke_test" in manifest["dataset_types"]
    summary = {"sample_count": len(manifest["records"]), "failed_count": 0,
               "manifest_sha256": manifest["sha256"], "checkpoint_sha256": checkpoint_hash,
               "checkpoint_best_epoch": meta["epoch"], "train_manifest_sha256": meta["train_manifest_sha256"],
               "val_manifest_sha256": meta["val_manifest_sha256"], "heldout_groups_checked": True,
               "grouping_units": {"train": meta["train_grouping_units"], "val": meta["val_grouping_units"], "test": manifest["grouping_units"]},
               "dataset_types": manifest["dataset_types"], "synthetic": synthetic,
               "interpretation": "Synthetic smoke test only; not real-world AI accuracy" if synthetic else "Manifest group IDs and image paths are disjoint; scene independence depends on grouping semantics. Official fold IDs alone do not establish scene independence.",
               "preprocessing": PREPROCESSING, "cnn_input_size": input_size,
               "classical_parameters": {"p": 6, "sigma": 1, "near_clip": 1.0},
               "timing_note": "Single-run diagnostics, no warmup. Decode excluded. CNN includes mask/weighted resize/exposure normalization/device transfer plus synchronized forward; classical methods use native resolution with their own mask validation. Different scales and work; not a fair speed benchmark.",
               "methods": {}}
    for method in ("tiny_cnn",) + METHODS:
        selected = [row for row in predictions if row["method"] == method]
        summary["methods"][method] = {"count": len(selected), **{
            metric: _stats([row[metric] for row in selected])
            for metric in ("recovery_error_deg", "reproduction_error_deg", "runtime_ms")}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    _json(output_dir / "summary.json", summary)
    return summary


def predict(checkpoint, image_path, black_level, white_level, mask_path=None, output=None,
            device="cpu", allow_synthetic=False):
    device = _device(device)
    model, payload, digest = load_checkpoint(checkpoint, allow_synthetic, device)
    image = load_rgb(Path(image_path), "linear", black_level, white_level)
    mask = load_mask(Path(mask_path), image.shape[:2]) if mask_path else None
    tensor = prepare_input(image, mask, payload["config"]["input_size"]).unsqueeze(0).to(device)
    with torch.inference_mode():
        light = model(tensor)[0].cpu().numpy().astype(float)
    if output:
        save_rgb(Path(output), apply_white_balance(image, light))
    meta, path = payload["metadata"], canonical_path(image_path)
    return {"illuminant_l2_rgb": light.tolist(), "green_normalized_gains": (light[1] / light).tolist(),
            "output": str(output) if output else None,
            "checkpoint_sha256": digest, "checkpoint_best_epoch": meta["epoch"],
            "preprocessing": PREPROCESSING, "input_size": payload["config"]["input_size"],
            "dataset_types": meta["dataset_types"], "synthetic": meta["synthetic"],
            "seen_train_image": path in meta["train_image_paths"], "seen_validation_image": path in meta["val_image_paths"],
            "interpretation": "Synthetic-trained smoke-test model; not real-data AWB performance" if meta["synthetic"] else "Learned global camera-RGB illuminant; calibration and camera generalization require validation"}
