"""Training and inference workflows"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageTensorDataset, Sample, generate_supports, index_dataset
from .detector import Flownomaly
from .protocol import PROTOCOL
from .progress import RunLogger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def result_directory(output_root: str, dataset: str, shot: int) -> Path:
    unit = "sample" if dataset == "realiad" else "image"
    return Path(output_root).expanduser().resolve() / dataset / f"{shot}-{unit}-shot"


def checkpoint_path(
    output_root: str, dataset: str, shot: int, seed: int, category: str
) -> Path:
    return (
        result_directory(output_root, dataset, shot)
        / "checkpoints"
        / f"seed={seed}"
        / f"{category}.pt"
    )


def _load_support_images(
    model: Flownomaly, samples: list[Sample], workers: int
) -> torch.Tensor:
    loader = DataLoader(
        ImageTensorDataset(samples, model.transform),
        batch_size=len(samples),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=4 if workers > 0 else None,
    )
    images, _indices = next(iter(loader))
    return images


def train_main_results(
    dataset: str,
    data_root: str,
    output_root: str,
    shot: int,
    seeds: list[int],
    workers: int,
    *,
    log_interval: float = 10.0,
    log_dir: str | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    with (
        run_logger(
            "train",
            dataset,
            shot,
            seeds,
            output_root,
            log_interval,
            log_dir,
            verbose,
            quiet,
        ) as reporter,
        reporter.stage("run", len(seeds), "categories", quiet=True) as run,
    ):
        with reporter.stage("dataset_indexing", split="train") as progress:
            indexed_train = index_dataset(dataset, data_root, "train")
            categories = sorted(indexed_train)
            progress.set(
                categories=len(categories), images=sum(map(len, indexed_train.values()))
            )
        if not categories:
            raise ValueError("dataset contains no training categories")
        run.total = len(seeds) * len(categories)
        reporter.emit(
            "run_plan",
            seeds=seeds,
            categories=len(categories),
            fits=run.total,
            workers=workers,
            output=result_directory(output_root, dataset, shot),
        )
        model = Flownomaly(reporter=reporter)
        model.load_encoder()

        for seed_index, seed in enumerate(seeds, 1):
            with (
                reporter.context(seed=seed),
                reporter.stage(
                    "seed",
                    len(categories),
                    "categories",
                    quiet=True,
                    seed_index=f"{seed_index}/{len(seeds)}",
                ) as seed_progress,
            ):
                seed_directory = checkpoint_path(
                    output_root, dataset, shot, seed, categories[0]
                ).parent
                existing = (
                    list(seed_directory.glob("*.pt")) if seed_directory.is_dir() else []
                )
                if existing:
                    raise FileExistsError(
                        f"refusing to mix a new seeded run with existing checkpoints in {seed_directory}"
                    )
                seed_everything(seed)
                with reporter.stage("support_selection", quiet=True):
                    supports = generate_supports(dataset, indexed_train, shot, seed)
                    seed_directory.mkdir(parents=True, exist_ok=False)
                for category_index, category in enumerate(categories, 1):
                    with (
                        reporter.context(category=category),
                        reporter.stage(
                            "category",
                            key=f"category:{category}",
                            category_index=f"{category_index}/{len(categories)}",
                        ) as category_progress,
                    ):
                        category_supports = supports[category]
                        expected = shot * 5 if dataset == "realiad" else shot
                        if len(category_supports) != expected:
                            raise RuntimeError(
                                f"{category}: expected {expected} support images, got {len(category_supports)}"
                            )
                        with reporter.stage(
                            "support_loading",
                            len(category_supports),
                            "images",
                            workers=workers,
                        ):
                            images = _load_support_images(
                                model, category_supports, workers
                            )
                        model.fit(images, category_supports, dataset, category)
                        path = checkpoint_path(
                            output_root, dataset, shot, seed, category
                        )
                        with reporter.stage("checkpoint_save", path=path):
                            model.save(path)
                        category_progress.set(
                            **model.fit_summary,
                            stage_times=reporter.phase_times(
                                category_progress.stage_id
                            ),
                        )
                    seed_progress.advance()
                    run.advance()


def run_logger(
    command: str,
    dataset: str,
    shot: int,
    seeds: list[int],
    output_root: str,
    interval: float,
    log_dir: str | None,
    verbose: bool,
    quiet: bool,
) -> RunLogger:
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return RunLogger(
        log_dir
        if log_dir is not None
        else result_directory(output_root, dataset, shot) / "logs",
        interval=interval,
        verbose=verbose,
        quiet=quiet,
        context={"command": command, "dataset": dataset, "shot": shot, "seeds": seeds},
        profile={
            "device": device,
            "torch": str(torch.__version__),
            "protocol": asdict(PROTOCOL),
            "command": command,
            "dataset": dataset,
            "shot": shot,
        },
    )


def _anomaly_groups(samples: list[Sample]) -> list[list[Sample]]:
    groups = []
    for anomaly_type in sorted({sample.anomaly_type for sample in samples}):
        group = [sample for sample in samples if sample.anomaly_type == anomaly_type]
        if not any(sample.view_index is not None for sample in group):
            group.sort(key=lambda item: item.stem)
        groups.append(group)
    return groups


def infer_main_results(
    dataset: str,
    data_root: str,
    output_root: str,
    shot: int,
    seeds: list[int],
    workers: int,
    *,
    log_interval: float = 10.0,
    log_dir: str | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    with (
        run_logger(
            "infer",
            dataset,
            shot,
            seeds,
            output_root,
            log_interval,
            log_dir,
            verbose,
            quiet,
        ) as reporter,
        reporter.stage("run", len(seeds), "categories", quiet=True) as run,
    ):
        with reporter.stage("dataset_indexing", split="test") as progress:
            indexed_test = index_dataset(dataset, data_root, "test")
            categories = sorted(indexed_test)
            progress.set(
                categories=len(categories), images=sum(map(len, indexed_test.values()))
            )
        if not categories:
            raise ValueError("dataset contains no inference categories")
        run.total = len(categories) * len(seeds)
        reporter.emit(
            "run_plan",
            seeds=seeds,
            categories=len(categories),
            workers=workers,
            output=result_directory(output_root, dataset, shot),
        )
        model = Flownomaly(reporter=reporter)
        model.load_encoder()
        map_root = result_directory(output_root, dataset, shot) / "anomaly_maps"
        map_root.mkdir(parents=True, exist_ok=True)

        for seed_index, seed in enumerate(seeds, 1):
            with (
                reporter.context(seed=seed),
                reporter.stage(
                    "seed",
                    len(categories),
                    "categories",
                    quiet=True,
                    seed_index=f"{seed_index}/{len(seeds)}",
                ) as seed_progress,
            ):
                output_path = map_root / f"seed={seed}.h5"
                if output_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite inference output: {output_path}"
                    )
                seed_everything(seed)
                temporary_path = output_path.with_suffix(".h5.incomplete")
                if temporary_path.exists():
                    temporary_path.unlink()
                try:
                    with h5py.File(temporary_path, "w") as output_file:
                        for category_index, category in enumerate(categories, 1):
                            with (
                                reporter.context(category=category),
                                reporter.stage(
                                    "category",
                                    key=f"category:{category}",
                                    category_index=f"{category_index}/{len(categories)}",
                                ) as category_progress,
                            ):
                                path = checkpoint_path(
                                    output_root, dataset, shot, seed, category
                                )
                                if not path.is_file():
                                    raise FileNotFoundError(
                                        f"missing checkpoint: {path}"
                                    )
                                with reporter.stage("checkpoint_load", path=path):
                                    model.load(path)
                                maps = []
                                with reporter.stage(
                                    "inference", len(indexed_test[category]), "images"
                                ) as progress:
                                    for samples in _anomaly_groups(
                                        indexed_test[category]
                                    ):
                                        loader = DataLoader(
                                            ImageTensorDataset(
                                                samples, model.transform
                                            ),
                                            batch_size=PROTOCOL.inference_batch_size,
                                            shuffle=False,
                                            num_workers=workers,
                                            pin_memory=True,
                                            prefetch_factor=4 if workers > 0 else None,
                                        )
                                        for images, _indices in loader:
                                            maps.append(model.score_batch(images))
                                            progress.advance(
                                                len(images),
                                                anomaly_type=samples[0].anomaly_type,
                                            )
                                category_maps = np.concatenate(maps, axis=0)
                                if len(category_maps) != len(indexed_test[category]):
                                    raise RuntimeError(
                                        f"inference lost images for {category}"
                                    )
                                with reporter.stage(
                                    "map_writing",
                                    images=len(category_maps),
                                    mib=category_maps.nbytes / 2**20,
                                ):
                                    output_file.create_dataset(
                                        category,
                                        data=category_maps,
                                        dtype="float32",
                                        compression="gzip",
                                        compression_opts=1,
                                    )
                                category_progress.set(
                                    images=len(category_maps),
                                    stage_times=reporter.phase_times(
                                        category_progress.stage_id
                                    ),
                                )
                            seed_progress.advance()
                            run.advance()
                        with reporter.stage("map_flush"):
                            output_file.flush()
                    with reporter.stage("map_commit", path=output_path):
                        os.replace(temporary_path, output_path)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()
