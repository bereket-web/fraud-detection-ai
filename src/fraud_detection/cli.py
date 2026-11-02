"""Command-line interface for the fraud-detection benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_detection.data import download_baf, load_baf, sha256_file, temporal_split
from fraud_detection.evaluate import run_benchmark


def _project_root() -> Path:
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-lab", description="Leakage-aware fraud detection benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download the version-pinned synthetic BAF Base variant.")
    download.add_argument("--destination", type=Path, default=Path("data/raw/base.csv"))

    run = subparsers.add_parser("run", help="Fit four approaches and report held-out metrics.")
    run.add_argument("--data", type=Path, default=Path("data/raw/base.csv"))
    run.add_argument("--output", type=Path, default=Path("reports"))
    run.add_argument("--max-train-rows", type=int, default=250_000, help="Use 0 to train on all earlier months.")
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--mlp-epochs", type=int, default=4)
    run.add_argument("--mlp-steps-per-epoch", type=int, default=60)
    run.add_argument("--anomaly-train-rows", type=int, default=100_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        path = download_baf(args.destination)
        print(f"Saved BAF Base.csv to {path}")
        print(f"SHA-256: {sha256_file(path)}")
        return

    if not args.data.exists():
        raise SystemExit(f"Dataset not found at {args.data}. Run `fraud-lab download` first.")
    frame = load_baf(args.data)
    splits = temporal_split(frame, max_train_rows=args.max_train_rows, seed=args.seed)
    summary = run_benchmark(
        splits,
        args.output,
        seed=args.seed,
        mlp_epochs=args.mlp_epochs,
        mlp_steps_per_epoch=args.mlp_steps_per_epoch,
        anomaly_train_rows=args.anomaly_train_rows,
        dataset_sha256=sha256_file(args.data),
    )
    print(json.dumps({"partitions": summary["partitions"], "test_metrics": summary["test_metrics"]}, indent=2))
    print(f"Reports written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
