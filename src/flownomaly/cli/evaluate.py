from __future__ import annotations

import argparse

from flownomaly.evaluation import evaluate_main_results
from flownomaly.protocol import DATASETS, SEEDS, SHOTS, validate_main_result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Flownomaly main results.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--shot", required=True, type=int, choices=SHOTS)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_main_result(args.dataset, args.shot, args.seeds)
    evaluate_main_results(
        args.dataset, args.data_root, args.output_root, args.shot, args.seeds
    )


if __name__ == "__main__":
    main()
