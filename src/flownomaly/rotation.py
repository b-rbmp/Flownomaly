"""Deterministic support rotations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import random


@dataclass(frozen=True)
class RotationView:
    image_key: str
    namespace: str
    augmentation_index: int
    rotation_index: int
    angle_degrees: float


def image_key_hash(image_key: str) -> int:
    key = "/".join(os.path.normpath(image_key).split(os.sep)[-4:])
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)


def rotation_view(
    image_key: str,
    namespace: str,
    augmentation_index: int,
    augmentation_seed: int,
    rotation_offset: int,
    maximum_angle: float,
) -> RotationView:
    rotation_index = int(rotation_offset) + int(augmentation_index)
    if namespace == "train" and augmentation_seed == 0 and rotation_offset == 0:
        private_seed = image_key_hash(image_key) + int(augmentation_index)
    else:
        material = (
            f"{image_key_hash(image_key)}\0{namespace}\0{augmentation_index}\0"
            f"{augmentation_seed}\0{rotation_index}"
        ).encode("utf-8")
        private_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    angle = random.Random(private_seed).uniform(0.0, float(maximum_angle))
    return RotationView(
        image_key=os.path.normpath(image_key),
        namespace=namespace,
        augmentation_index=int(augmentation_index),
        rotation_index=rotation_index,
        angle_degrees=angle,
    )
