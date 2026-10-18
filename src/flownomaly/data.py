"""Dataset indexing and paper-seeded support selection."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from collections.abc import Callable

from PIL import Image
from torch.utils.data import Dataset


REAL_IAD_PATTERN = re.compile(
    r"(?P<sample>[a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_"
    r"C(?P<view>[0-9]+)_"
)


@dataclass(frozen=True)
class Sample:
    category: str
    split: str
    anomaly_type: str
    image_path: str
    image_key: str
    mask_path: str | None
    stem: str
    sample_id: str
    view_index: int | None = None


class ImageTensorDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        with Image.open(self.samples[index].image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def _folder_samples(root: Path, category: str, split: str) -> list[Sample]:
    split_root = root / category / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing dataset directory: {split_root}")
    samples: list[Sample] = []
    for anomaly_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        anomaly_type = anomaly_dir.name
        for image_path in sorted(anomaly_dir.iterdir()):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            mask = None
            if split == "test" and anomaly_type != "good":
                mask = (
                    root
                    / category
                    / "ground_truth"
                    / anomaly_type
                    / (image_path.stem + "_mask.png")
                )
                if not mask.is_file():
                    raise FileNotFoundError(f"missing MVTec-AD mask: {mask}")
            key = f"{category}/{split}/{anomaly_type}/{image_path.name}"
            samples.append(
                Sample(
                    category=category,
                    split=split,
                    anomaly_type=anomaly_type,
                    image_path=str(image_path.resolve()),
                    image_key=key,
                    mask_path=str(mask.resolve()) if mask is not None else None,
                    stem=image_path.stem,
                    sample_id=image_path.stem,
                )
            )
    return samples


def _visa_csv_root(root: Path) -> Path | None:
    if (root / "split_csv" / "1cls.csv").is_file():
        return root
    return None


def _visa_raw_samples(root: Path, split: str) -> dict[str, list[Sample]]:
    csv_path = root / "split_csv" / "1cls.csv"
    categories: dict[str, list[Sample]] = {}
    with csv_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if row["split"] != split:
                continue
            category = row["object"]
            anomaly_type = "good" if row["label"] == "normal" else "bad"
            image_path = (root / row["image"]).resolve()
            mask_value = row.get("mask", "").strip()
            mask_path = (root / mask_value).resolve() if mask_value else None
            if not image_path.is_file():
                raise FileNotFoundError(f"missing VisA image: {image_path}")
            if mask_path is not None and not mask_path.is_file():
                raise FileNotFoundError(f"missing VisA mask: {mask_path}")
            key = f"{category}/{split}/{anomaly_type}/{image_path.name}"
            categories.setdefault(category, []).append(
                Sample(
                    category=category,
                    split=split,
                    anomaly_type=anomaly_type,
                    image_path=str(image_path),
                    image_key=key,
                    mask_path=str(mask_path) if mask_path is not None else None,
                    stem=image_path.stem,
                    sample_id=image_path.stem,
                )
            )
    for samples in categories.values():
        samples.sort(key=lambda item: (item.anomaly_type, item.stem))
    return categories


def _visa_processed_root(root: Path) -> Path:
    candidate = root / "1cls"
    return candidate if candidate.is_dir() else root


def _visa_processed_samples(root: Path, split: str) -> dict[str, list[Sample]]:
    base = _visa_processed_root(root)
    categories: dict[str, list[Sample]] = {}
    for category_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        split_dir = category_dir / split
        if not split_dir.is_dir():
            continue
        samples: list[Sample] = []
        for anomaly_dir in sorted(
            path for path in split_dir.iterdir() if path.is_dir()
        ):
            for image_path in sorted(anomaly_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                anomaly_type = anomaly_dir.name
                mask = None
                if split == "test" and anomaly_type != "good":
                    mask = (
                        category_dir
                        / "ground_truth"
                        / anomaly_type
                        / (image_path.stem + ".png")
                    )
                    if not mask.is_file():
                        raise FileNotFoundError(f"missing VisA mask: {mask}")
                key = f"{category_dir.name}/{split}/{anomaly_type}/{image_path.name}"
                samples.append(
                    Sample(
                        category=category_dir.name,
                        split=split,
                        anomaly_type=anomaly_type,
                        image_path=str(image_path.resolve()),
                        image_key=key,
                        mask_path=str(mask.resolve()) if mask is not None else None,
                        stem=image_path.stem,
                        sample_id=image_path.stem,
                    )
                )
        categories[category_dir.name] = samples
    return categories


def _real_iad_manifest_dir(root: Path) -> Path:
    for candidate in (root / "realiad_jsons", root / "realiad_jsons" / "realiad_jsons"):
        if candidate.is_dir() and any(candidate.glob("*.json")):
            return candidate
    raise FileNotFoundError(
        "Real-IAD category JSON files were not found under realiad_jsons/"
    )


def _resolve_real_iad_path(root: Path, category: str, prefix: str, value: str) -> Path:
    relative = Path(value)
    candidates = (
        root / "realiad_1024" / prefix / relative,
        root / prefix / relative,
        root / "realiad_1024" / category / relative,
        root / "realiad_1024" / relative,
        root / relative,
    )
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Real-IAD file not found: {value!r}")


def _real_iad_samples(root: Path, category: str, split: str) -> list[Sample]:
    manifest_path = _real_iad_manifest_dir(root) / f"{category}.json"
    with manifest_path.open(encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    prefix = str(manifest["meta"]["prefix"]).strip("/")
    normal_class = str(manifest["meta"]["normal_class"])
    samples: list[Sample] = []
    for record in manifest[split]:
        image_path = _resolve_real_iad_path(
            root, category, prefix, str(record["image_path"])
        )
        is_normal = (
            split == "train"
            or record["mask_path"] in (None, "")
            or str(record["anomaly_class"]) == normal_class
        )
        anomaly_type = "good" if is_normal else str(record["anomaly_class"])
        mask_path = None
        if not is_normal:
            mask_path = _resolve_real_iad_path(
                root, category, prefix, str(record["mask_path"])
            )
        match = REAL_IAD_PATTERN.search(image_path.stem)
        if match is None:
            raise ValueError(f"cannot parse Real-IAD sample/view: {image_path.name}")
        try:
            key = image_path.relative_to(root.resolve()).as_posix()
        except ValueError:
            key = "/".join(image_path.parts[-4:])
        samples.append(
            Sample(
                category=category,
                split=split,
                anomaly_type=anomaly_type,
                image_path=str(image_path),
                image_key=key,
                mask_path=str(mask_path) if mask_path is not None else None,
                stem=image_path.stem,
                sample_id=match.group("sample"),
                view_index=int(match.group("view")),
            )
        )
    return samples


def index_dataset(dataset: str, data_root: str, split: str) -> dict[str, list[Sample]]:
    root = Path(data_root).expanduser().resolve()
    if dataset == "mvtec":
        categories = sorted(path.name for path in root.iterdir() if path.is_dir())
        return {
            category: _folder_samples(root, category, split) for category in categories
        }
    if dataset == "visa":
        if _visa_csv_root(root) is not None:
            return _visa_raw_samples(root, split)
        return _visa_processed_samples(root, split)
    if dataset == "realiad":
        categories = sorted(
            path.stem for path in _real_iad_manifest_dir(root).glob("*.json")
        )
        return {
            category: _real_iad_samples(root, category, split)
            for category in categories
        }
    raise ValueError(f"unsupported dataset: {dataset!r}")


def _real_iad_selection_seed(seed: int, category: str) -> int:
    material = f"real-iad-support-v1\0{seed}\0{category}\0sample"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def generate_supports(
    dataset: str,
    indexed_train: dict[str, list[Sample]],
    shot: int,
    seed: int,
) -> dict[str, list[Sample]]:
    """Recreate the support draws used by the main benchmark pipeline."""
    selected: dict[str, list[Sample]] = {}
    if dataset in {"mvtec", "visa"}:
        rng = random.Random(seed)
        for category in sorted(indexed_train):
            candidates = sorted(
                (
                    sample
                    for sample in indexed_train[category]
                    if sample.anomaly_type == "good"
                ),
                key=lambda item: item.image_path,
            )
            indices = list(range(len(candidates)))
            rng.shuffle(indices)
            if shot > len(indices):
                raise ValueError(f"{category} has fewer than {shot} normal images")
            selected[category] = [candidates[index] for index in indices[:shot]]
        return selected

    for category in sorted(indexed_train):
        groups: dict[str, list[Sample]] = {}
        for sample in indexed_train[category]:
            groups.setdefault(sample.sample_id, []).append(sample)
        for views in groups.values():
            views.sort(key=lambda item: (item.view_index, item.stem))
            if len(views) != 5:
                raise ValueError(
                    f"Real-IAD sample {views[0].sample_id!r} does not contain five views"
                )
        sample_ids = sorted(groups)
        rng = random.Random(_real_iad_selection_seed(seed, category))
        rng.shuffle(sample_ids)
        if shot > len(sample_ids):
            raise ValueError(f"{category} has fewer than {shot} physical samples")
        selected[category] = [
            view for sample_id in sample_ids[:shot] for view in groups[sample_id]
        ]
    return selected


def evaluation_order(samples: list[Sample]) -> list[Sample]:
    """Match inference's sorted anomaly-type traversal."""
    ordered: list[Sample] = []
    for anomaly_type in sorted({sample.anomaly_type for sample in samples}):
        group = [sample for sample in samples if sample.anomaly_type == anomaly_type]
        if (
            group
            and group[0].category
            and group[0].split
            and not any(sample.view_index is not None for sample in group)
        ):
            group.sort(key=lambda item: item.stem)
        ordered.extend(group)
    return ordered
