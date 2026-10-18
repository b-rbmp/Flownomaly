from __future__ import annotations

import argparse

from flownomaly.protocol import DATASETS, SEEDS, SHOTS, validate_main_result
from flownomaly.workflow import train_main_results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit Flownomaly main-result checkpoints."
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--shot", required=True, type=int, choices=SHOTS)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS
    )
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_main_result(args.dataset, args.shot, args.seeds)
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    train_main_results(
        args.dataset,
        args.data_root,
        args.output_root,
        args.shot,
        args.seeds,
        args.workers,
    )


if __name__ == "__main__":
    main()
