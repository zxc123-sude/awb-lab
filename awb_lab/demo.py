"""Deterministic synthetic fixtures: pipeline smoke tests, NOT real benchmarks."""

import csv
from pathlib import Path

import numpy as np

from .image_io import save_rgb


def make_demo(output_dir: Path) -> Path:
    """Create three linear RGB NPYs, valid masks, display previews and manifest.

    A neutral textured surface under a known diagonal illuminant is deliberately
    easy for these estimators. A corner patch is explicitly excluded by a mask.
    This checks units, masks, gains and I/O; it demonstrates no real-world rank.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:96, :128]
    neutral = 0.12 + 0.20 * x / 127 + 0.10 * y / 95 + 0.025 * np.sin(x / 5) * np.cos(y / 7)
    lights = ((1.5, 1.0, 0.65), (0.75, 1.0, 1.45), (1.1, 1.0, 0.9))
    rows = []
    for index, light in enumerate(lights, 1):
        image = neutral[..., None] * np.array(light)
        mask = np.ones(neutral.shape, dtype=np.uint8)
        image[:16, :16] = (0.8, 0.1, 0.2)
        mask[:16, :16] = 0
        image_name = f"synthetic_{index}.npy"
        mask_name = f"synthetic_{index}_mask.npy"
        np.save(str(output_dir / image_name), image, allow_pickle=False)
        np.save(str(output_dir / mask_name), mask, allow_pickle=False)
        save_rgb(output_dir / f"synthetic_{index}_preview.png", image)
        rows.append({
            "sample_id": f"synthetic_{index}", "image": image_name,
            "illuminant_r": light[0], "illuminant_g": light[1], "illuminant_b": light[2],
            "black_level": 0, "white_level": 1, "mask": mask_name,
            "camera": "synthetic_diagonal_model", "split": "smoke_test",
            "dataset_type": "synthetic_smoke_test",
        })
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "SYNTHETIC_ONLY.txt").write_text(
        "Synthetic neutral surfaces under known diagonal illuminants.\n"
        "Smoke-test fixtures only; not a real camera dataset or a benchmark claim.\n"
        "The valid mask excludes the artificial corner color patch.\n",
        encoding="utf-8",
    )
    return manifest
