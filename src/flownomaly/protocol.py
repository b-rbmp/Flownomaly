"""Fixed constants for the published few-shot protocol."""

from __future__ import annotations

from dataclasses import dataclass


DATASETS = ("mvtec", "visa", "realiad")
SHOTS = (1, 2, 4)
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class PublishedProtocol:
    backbone: str = "dinov2_vitg14_reg"
    image_size: int = 672
    patch_size: int = 14
    target_layers: tuple[int, ...] = (15, 19, 23, 27, 31) # Note: these are 0-indexed layer numbers, not 1-indexed, therefore corresponds to the 16th, 20th, 24th, 28th, and 32nd layers of the model.
    extraction_batch_size: int = 8
    inference_batch_size: int = 128

    training_rotations: int = 30
    validation_rotations: int = 3
    calibration_rotations: int = 3
    maximum_rotation_angle: float = 345.0
    training_rotation_seed: int = 0
    validation_seed: int = 1234
    validation_rotation_offset: int = 100000
    calibration_seed: int = 5678
    calibration_rotation_offset: int = 200000

    bottleneck_dim: int = 768
    residual_blocks: int = 4
    mlp_ratio: int = 4
    time_embedding_dim: int = 256

    flow_batch_size: int = 10240
    maximum_epochs: int = 1000
    learning_rate: float = 1.0e-3
    minimum_learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-4
    validation_frequency: int = 3
    early_stopping_patience: int = 10
    validation_minimum_delta: float = 1.0e-4

    jgn_candidates: tuple[float, ...] = tuple(i / 100 for i in range(5, 100, 5))
    jgn_noise_pairs: int = 4
    jgn_probes: int = 4
    jgn_token_fraction: float = 0.01
    jgn_epsilon: float = 1.0e-8
    jgn_batch_size: int = 4096

    localization_sigma_tokens: float = 0.8
    image_top_fraction: float = 0.01
    metric_strips: int = 200
    aupro_fpr_limit: float = 0.3

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size


PROTOCOL = PublishedProtocol()


def validate_main_result(dataset: str, shot: int, seeds: list[int]) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    if shot not in SHOTS:
        raise ValueError(f"shot must be one of {SHOTS}, got {shot}")
    if not seeds or any(seed not in SEEDS for seed in seeds):
        raise ValueError(f"seeds must be drawn from {SEEDS}")
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seeds are not allowed")
