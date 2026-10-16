"""Optional trainable AWB command-line workflow."""

import argparse
import json
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="Train a compact CNN for global AWB from paired linear RGB; no pretrained weights.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate independent synthetic smoke-test splits, not a benchmark dataset")
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.add_argument("--train-count", type=int, default=24)
    demo.add_argument("--val-count", type=int, default=8)
    demo.add_argument("--test-count", type=int, default=8)
    demo.add_argument("--seed", type=int, default=42)
    training = commands.add_parser("train", help="Train only on TRAIN; select best.pt only on VAL (disjoint groups required)")
    training.add_argument("train_csv", type=Path)
    training.add_argument("val_csv", type=Path)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--input-size", type=int, default=128)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    evaluation = commands.add_parser("evaluate", help="Held-out group evaluation of CNN plus four classical methods")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("test_csv", type=Path)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    prediction = commands.add_parser("predict", help="Predict a global camera-RGB illuminant from LINEAR RGB only")
    prediction.add_argument("checkpoint", type=Path)
    prediction.add_argument("input", type=Path)
    prediction.add_argument("--black-level", type=float, required=True, help="Original file units; explicitly 0 if already corrected")
    prediction.add_argument("--white-level", type=float, required=True, help="Original file numeric saturation, never assumed from dtype")
    prediction.add_argument("--mask", type=Path)
    prediction.add_argument("--output", type=Path, help="Optional corrected .npy linear or .png/.jpg display preview")
    for command in (training, evaluation, prediction):
        command.add_argument("--device", default="cpu", help="cpu (default) or available cuda device")
        command.add_argument("--allow-synthetic", action="store_true", help="Explicit opt-in to synthetic smoke-test data/models; not real-data evidence")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import make_demo
            result = make_demo(args.output_dir, args.train_count, args.val_count, args.test_count, args.seed)
        else:
            from .runner import evaluate, predict, train
            if args.command == "train":
                result = train(args.train_csv, args.val_csv, args.output_dir, args.epochs, args.batch_size,
                               args.input_size, args.seed, args.device, args.learning_rate, args.allow_synthetic)
            elif args.command == "evaluate":
                result = evaluate(args.checkpoint, args.test_csv, args.output_dir, args.device, args.allow_synthetic)
            else:
                result = predict(args.checkpoint, args.input, args.black_level, args.white_level, args.mask,
                                 args.output, args.device, args.allow_synthetic)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print('error: optional PyTorch dependency is missing; install the project AI extra: python -m pip install ".[ai]"', file=sys.stderr)
            return 2
        raise
    except (ValueError, OSError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
