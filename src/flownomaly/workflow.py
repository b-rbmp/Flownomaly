"""Training and inference workflows"""

from __future__ import annotations

import os
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageTensorDataset, Sample, generate_supports, index_dataset
from .detector import Flownomaly
from .protocol import PROTOCOL


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
) -> None:
    indexed_train = index_dataset(dataset, data_root, "train")
    categories = sorted(indexed_train)
    model = Flownomaly()
    model.load_encoder()

    for seed in seeds:
        seed_directory = checkpoint_path(
            output_root, dataset, shot, seed, categories[0]
        ).parent
        existing = list(seed_directory.glob("*.pt")) if seed_directory.is_dir() else []
        if existing:
            raise FileExistsError(
                f"refusing to mix a new seeded run with existing checkpoints in {seed_directory}"
            )
        seed_everything(seed)
        supports = generate_supports(dataset, indexed_train, shot, seed)
        seed_directory.mkdir(parents=True, exist_ok=False)
        for category in categories:
            category_supports = supports[category]
            expected = shot * 5 if dataset == "realiad" else shot
            if len(category_supports) != expected:
                raise RuntimeError(
                    f"{category}: expected {expected} support images, got {len(category_supports)}"
                )
            print(
                f"train dataset={dataset} shot={shot} seed={seed} "
                f"category={category} supports={len(category_supports)}",
                flush=True,
            )
            images = _load_support_images(model, category_supports, workers)
            model.fit(images, category_supports, dataset, category)
            model.save(checkpoint_path(output_root, dataset, shot, seed, category))


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
) -> None:
    indexed_test = index_dataset(dataset, data_root, "test")
    categories = sorted(indexed_test)
    model = Flownomaly()
    model.load_encoder()
    map_root = result_directory(output_root, dataset, shot) / "anomaly_maps"
    map_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
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
                for category in categories:
                    path = checkpoint_path(output_root, dataset, shot, seed, category)
                    if not path.is_file():
                        raise FileNotFoundError(f"missing checkpoint: {path}")
                    model.load(path)
                    maps = []
                    for samples in _anomaly_groups(indexed_test[category]):
                        loader = DataLoader(
                            ImageTensorDataset(samples, model.transform),
                            batch_size=PROTOCOL.inference_batch_size,
                            shuffle=False,
                            num_workers=workers,
                            pin_memory=True,
                            prefetch_factor=4 if workers > 0 else None,
                        )
                        for images, _indices in loader:
                            maps.append(model.score_batch(images))
                    category_maps = np.concatenate(maps, axis=0)
                    if len(category_maps) != len(indexed_test[category]):
                        raise RuntimeError(f"inference lost images for {category}")
                    output_file.create_dataset(
                        category,
                        data=category_maps,
                        dtype="float32",
                        compression="gzip",
                        compression_opts=1,
                    )
                    print(
                        f"infer dataset={dataset} shot={shot} seed={seed} "
                        f"category={category}",
                        flush=True,
                    )
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
