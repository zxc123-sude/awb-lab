"""Command-line entry point for demonstration correction and paired evaluation."""

import argparse
import json
import sys
from pathlib import Path

from .algorithms import METHODS, apply_white_balance, estimate_illuminant
from .demo import make_demo
from .evaluation import evaluate
from .image_io import load_mask, load_rgb, save_rgb


def _parameters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--p", type=float, default=6.0, help="Minkowski exponent >= 1 (Shades of Gray / Gray Edge)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gray Edge Gaussian sigma in pixels")
    parser.add_argument("--near-clip", type=float, default=1.0,
                        help="Exclude pixels with any normalized channel >= threshold (default 1.0; evaluation requires (0,1])")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical AWB on linear RGB; sRGB correction is a visual demo.")
    commands = parser.add_subparsers(dest="command", required=True)
    correct = commands.add_parser("correct", help="Estimate light and apply green-normalized gains")
    correct.add_argument("input", type=Path)
    correct.add_argument("--output", type=Path, required=True, help=".npy linear values, or .png/.jpg 8-bit sRGB preview")
    correct.add_argument("--method", choices=METHODS, default="gray_world")
    correct.add_argument("--input-space", choices=("srgb", "linear"), default="srgb")
    correct.add_argument("--black-level", type=float, default=0.0, help="Black level in original file numeric units")
    correct.add_argument("--white-level", type=float, help="White level in original file units; required for linear RGB")
    correct.add_argument("--mask", type=Path, help="H x W mask; nonzero=valid")
    _parameters(correct)
    evaluation = commands.add_parser("evaluate", help="Evaluate paired LINEAR RGB manifest; JPEG is refused")
    evaluation.add_argument("manifest", type=Path)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    _parameters(evaluation)
    demo = commands.add_parser("demo", help="Generate deterministic synthetic smoke-test fixtures, not benchmark data")
    demo.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            manifest = make_demo(args.output_dir)
            print(f"Synthetic smoke-test data only (not real benchmark data): {manifest}")
        elif args.command == "correct":
            image = load_rgb(args.input, args.input_space, args.black_level, args.white_level)
            mask = load_mask(args.mask, image.shape[:2]) if args.mask else None
            light = estimate_illuminant(image, args.method, mask, args.p, args.sigma, args.near_clip)
            save_rgb(args.output, apply_white_balance(image, light))
            print(json.dumps({
                "method": args.method, "illuminant_l2_rgb": light.tolist(),
                "green_normalized_gains": (light[1] / light).tolist(),
                "output": str(args.output),
                "interpretation": "sRGB visual demo; not camera-RAW benchmark" if args.input_space == "srgb" else "linear RGB correction",
                "output_encoding": "linear float64 (unclipped)" if args.output.suffix.lower() == ".npy" else "8-bit sRGB preview (clipped)",
            }, indent=2, allow_nan=False))
        else:
            summary = evaluate(args.manifest, args.output_dir, args.methods, args.p, args.sigma, args.near_clip)
            print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
