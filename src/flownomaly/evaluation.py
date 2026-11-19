"""Evaluation of native Flownomaly anomaly maps."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional

from .data import Sample, evaluation_order, index_dataset
from .metrics import PixelMetrics, image_metrics
from .protocol import PROTOCOL
from .progress import DEFAULT_LOG_INTERVAL, RunLogger, get_reporter
from .workflow import result_directory, run_logger


def _upsample(raw_map: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    tensor = torch.from_numpy(np.ascontiguousarray(raw_map))[None, None]
    return (
        functional.interpolate(
            tensor, size=(height, width), mode="bilinear", align_corners=False
        )[0, 0]
        .numpy()
        .astype(np.float32, copy=False)
    )


def localization_map(raw_map: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    smoothed = cv2.GaussianBlur(
        np.asarray(raw_map, dtype=np.float32),
        ksize=(0, 0),
        sigmaX=PROTOCOL.localization_sigma_tokens,
        sigmaY=PROTOCOL.localization_sigma_tokens,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return _upsample(smoothed, target_size)


def top_one_percent_score(raw_map: np.ndarray, target_size: tuple[int, int]) -> float:
    values = _upsample(raw_map, target_size).reshape(-1)
    count = max(1, int(len(values) * PROTOCOL.image_top_fraction))
    if count < len(values):
        values = np.partition(values, len(values) - count)[len(values) - count :]
    return float(np.mean(np.sort(values)[::-1]))


def _mask(sample: Sample, shape: tuple[int, int]) -> np.ndarray:
    if sample.mask_path is None:
        return np.zeros(shape, dtype=np.uint8)
    with Image.open(sample.mask_path) as image:
        mask = np.asarray(image.convert("L"))
    if mask.shape != shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match prediction shape {shape}: {sample.mask_path}"
        )
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _evaluate_category(
    raw_maps, samples: list[Sample], *, reporter: RunLogger | None = None
) -> dict:
    reporter = reporter if reporter is not None else get_reporter()
    if len(raw_maps) != len(samples):
        raise ValueError("stored map count does not match the dataset index")
    image_scores: list[float] = []
    labels: list[int] = []
    lower, upper = float("inf"), float("-inf")
    target_sizes = []
    with reporter.stage(
        "evaluation",
        len(samples),
        "images",
        key="evaluation_scores",
        display="Evaluation",
        pass_name="1/2 image scores",
    ) as progress:
        for raw_map, sample in zip(raw_maps, samples, strict=True):
            with Image.open(sample.image_path) as image:
                target_size = image.size
            target_sizes.append(target_size)
            full_map = localization_map(raw_map, target_size)
            lower = min(lower, float(full_map.min()))
            upper = max(upper, float(full_map.max()))
            image_scores.append(top_one_percent_score(raw_map, target_size))
            labels.append(0 if sample.anomaly_type == "good" else 1)
            progress.advance()

        progress.set_work(len(samples), "images", key="evaluation_pixels")
        progress.set(pass_name="2/2 pixel metrics")
        with reporter.stage("pixel_metric_initialization"):
            pixel_metrics = PixelMetrics(lower, upper)
        for raw_map, sample, target_size in zip(
            raw_maps, samples, target_sizes, strict=True
        ):
            full_map = localization_map(raw_map, target_size)
            pixel_metrics.add(full_map, _mask(sample, full_map.shape))
            progress.advance()

        progress.set_work(1, "steps", key="evaluation_finalize")
        progress.set(pass_name="final metrics")
        with reporter.stage("image_metrics"):
            image_result = image_metrics(image_scores, labels)
        with reporter.stage("pixel_metrics"):
            pixel_result = pixel_metrics.result()
        progress.set(
            image_auroc=image_result["auroc"], pixel_auroc=pixel_result["auroc"]
        )
        result = {"image": image_result, "pixel": pixel_result}
        reporter.emit("category_metrics", metrics=result)
    return result


def _mean_categories(categories: dict[str, dict]) -> dict:
    return {
        level: {
            metric: float(
                np.mean([values[level][metric] for values in categories.values()])
            )
            for metric in next(iter(categories.values()))[level]
        }
        for level in ("image", "pixel")
    }


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def evaluate_main_results(
    dataset: str,
    data_root: str,
    output_root: str,
    shot: int,
    seeds: list[int],
    *,
    log_interval: float = DEFAULT_LOG_INTERVAL,
    log_dir: str | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    with run_logger(
        "evaluate",
        dataset,
        shot,
        seeds,
        output_root,
        log_interval,
        log_dir,
        verbose,
        quiet,
    ) as reporter:
        with reporter.stage("dataset_indexing", split="test"):
            indexed = index_dataset(dataset, data_root, "test")
            ordered = {
                category: evaluation_order(samples)
                for category, samples in indexed.items()
            }
        if not ordered:
            raise ValueError("dataset contains no evaluation categories")
        run_directory = result_directory(output_root, dataset, shot)
        reporter.emit("run_plan", categories=len(ordered), output=run_directory)
        seed_results = []
        for seed in seeds:
            with reporter.context(seed=seed):
                map_path = run_directory / "anomaly_maps" / f"seed={seed}.h5"
                if not map_path.is_file():
                    raise FileNotFoundError(f"missing inference maps: {map_path}")
                categories = {}
                with h5py.File(map_path, "r") as input_file:
                    if sorted(input_file) != sorted(ordered):
                        raise RuntimeError(
                            "stored categories do not match the dataset index"
                        )
                    for category_index, category in enumerate(sorted(ordered), 1):
                        with reporter.context(
                            category=category,
                            category_index=f"{category_index}/{len(ordered)}",
                        ):
                            categories[category] = _evaluate_category(
                                input_file[category],
                                ordered[category],
                                reporter=reporter,
                            )
                result = {
                    "categories": categories,
                    "mean": _mean_categories(categories),
                }
                with reporter.stage("seed_metrics_writing"):
                    _write_json(run_directory / f"metrics_seed={seed}.json", result)
                reporter.emit("seed_metrics", metrics=result["mean"])
                seed_results.append(result)

        with reporter.stage("summary_writing"):
            summary = {"mean": {}}
            for level in ("image", "pixel"):
                summary["mean"][level] = {}
                for metric in seed_results[0]["mean"][level]:
                    values = np.asarray(
                        [result["mean"][level][metric] for result in seed_results],
                        dtype=np.float64,
                    )
                    summary["mean"][level][metric] = {
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    }
            _write_json(run_directory / "metrics_summary.json", summary)
            with (run_directory / "metrics_summary.md").open(
                "w", encoding="utf-8"
            ) as output:
                output.write("| Level | Metric | Mean | Std |\n|---|---:|---:|---:|\n")
                for level in ("image", "pixel"):
                    for metric, values in summary["mean"][level].items():
                        output.write(
                            f"| {level} | {metric} | {100 * values['mean']:.1f} | "
                            f"{100 * values['std']:.1f} |\n"
                        )
        reporter.emit(
            "evaluation_completed",
            console=True,
            metrics=summary["mean"],
            output=run_directory / "metrics_summary.json",
        )
