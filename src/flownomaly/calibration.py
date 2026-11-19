"""Normal-only Jacobian gain calibration for the operating time."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math

import torch

from .protocol import PROTOCOL
from .progress import RunLogger, Stage, get_reporter
from .rotation import image_key_hash


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take the median of an empty sequence")
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, 0.5).item())


def _seed_material(
    calibration_seed: int,
    category: str,
    view: dict,
    pair_index: int,
) -> list[str]:
    return [
        str(calibration_seed),
        category,
        str(view["view_id"]),
        str(image_key_hash(view["image_key"])),
        str(view["augmentation_index"]),
        str(view["rotation_index"]),
        str(pair_index),
    ]


def _private_seed(
    namespace: str, material: list[str], probe_index: int | None = None
) -> int:
    values = [namespace, *material]
    if probe_index is not None:
        values.append(str(probe_index))
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _residual(
    velocity_model,
    query: torch.Tensor,
    noise: torch.Tensor,
    time: float,
) -> torch.Tensor:
    query = query.float()
    noise = noise.float()
    state = float(time) * query + (1.0 - float(time)) * noise
    times = torch.full(
        (len(query), 1),
        float(time),
        device=query.device,
        dtype=torch.float32,
    )
    with torch.amp.autocast(device_type=query.device.type, enabled=False):
        velocity = velocity_model(state, times)
    recovery = state + (1.0 - float(time)) * velocity.float()
    return query - recovery


def select_operating_time(
    velocity_model,
    category: str,
    calibration_views: list[dict],
    device: torch.device,
    *,
    reporter: RunLogger | None = None,
    progress: Stage | None = None,
) -> float:
    """Return the JGN-selected operating time for one fitted category."""
    reporter = reporter if reporter is not None else get_reporter()
    if not calibration_views:
        raise ValueError("calibration requires at least one view")
    if any(len(view["features"]) == 0 for view in calibration_views):
        raise ValueError("calibration views should contain valid tokens")
    velocity_model.eval()
    for parameter in velocity_model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)

    candidate_values = list(PROTOCOL.jgn_candidates)
    view_scores: dict[float, list[float]] = {value: [] for value in candidate_values}

    total_jvps = (
        sum(
            math.ceil(len(view["features"]) / PROTOCOL.jgn_batch_size)
            for view in calibration_views
        )
        * len(candidate_values)
        * PROTOCOL.jgn_noise_pairs
        * PROTOCOL.jgn_probes
    )
    owns_progress = progress is None
    context = (
        reporter.stage(
            "calibration",
            total_jvps,
            "JVPs",
            key="calibration_search",
            display="Calibration",
        )
        if owns_progress
        else nullcontext(progress)
    )
    with context as progress:
        if not owns_progress:
            progress.set_work(total_jvps, "JVPs", key="calibration_search")
        for view in calibration_views:
            features = view["features"].detach().cpu().float().contiguous()
            token_count, feature_dimension = features.shape
            top_count = max(1, int(token_count * PROTOCOL.jgn_token_fraction))
            pair_ratios: dict[float, list[float]] = {
                value: [] for value in candidate_values
            }

            for pair_index in range(PROTOCOL.jgn_noise_pairs):
                material = _seed_material(
                    PROTOCOL.calibration_seed, category, view, pair_index
                )
                noise_generator = torch.Generator(device="cpu")
                noise_generator.manual_seed(_private_seed("jvp_noise_v1", material))
                noise = torch.randn(
                    features.shape, generator=noise_generator, dtype=torch.float32
                )
                probes = []
                for probe_index in range(PROTOCOL.jgn_probes):
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(
                        _private_seed("jvp_rademacher_v1", material, probe_index)
                    )
                    probe = torch.randint(
                        0,
                        2,
                        features.shape,
                        generator=generator,
                        dtype=torch.int8,
                    ).float()
                    probes.append(
                        probe.mul_(2.0).sub_(1.0).div_(math.sqrt(feature_dimension))
                    )

                for candidate in candidate_values:
                    residual_scores: list[torch.Tensor] = []
                    probe_sums = [0.0] * PROTOCOL.jgn_probes
                    for start in range(0, token_count, PROTOCOL.jgn_batch_size):
                        stop = min(start + PROTOCOL.jgn_batch_size, token_count)
                        query_chunk = features[start:stop].to(device)
                        noise_chunk = noise[start:stop].to(device)
                        for probe_index, probe in enumerate(probes):
                            residual, residual_jvp = torch.func.jvp(
                                lambda query, fixed_noise=noise_chunk, fixed_time=candidate: (
                                    _residual(
                                        velocity_model, query, fixed_noise, fixed_time
                                    )
                                ),
                                (query_chunk,),
                                (probe[start:stop].to(device),),
                            )
                            if probe_index == 0:
                                residual_scores.append(
                                    residual.detach()
                                    .cpu()
                                    .double()
                                    .square()
                                    .sum(dim=-1)
                                )
                            probe_sums[probe_index] += float(
                                residual_jvp.detach()
                                .cpu()
                                .double()
                                .square()
                                .sum()
                                .item()
                            )
                            progress.advance()
                    gains = [value / token_count for value in probe_sums]
                    gain = sum(gains) / PROTOCOL.jgn_probes
                    scores = torch.cat(residual_scores)
                    normal_floor = float(
                        torch.topk(scores, top_count, largest=True, sorted=False)
                        .values.mean()
                        .item()
                    )
                    ratio = gain / (normal_floor + PROTOCOL.jgn_epsilon)
                    if not all(
                        math.isfinite(value) for value in (gain, normal_floor, ratio)
                    ):
                        raise FloatingPointError(
                            f"non-finite JGN score for category={category}, "
                            f"view={view['view_id']}, time={candidate}: "
                            f"gain={gain}, normal_floor={normal_floor}, score={ratio}"
                        )
                    pair_ratios[candidate].append(ratio)
                    reporter.emit(
                        "calibration_candidate",
                        level="DEBUG",
                        view_id=view["view_id"],
                        noise_pair=pair_index + 1,
                        candidate_time=candidate,
                        gain=gain,
                        normal_floor=normal_floor,
                        score=ratio,
                        valid_tokens=token_count,
                        top_tokens=top_count,
                    )
            for candidate in candidate_values:
                view_scores[candidate].append(_median(pair_ratios[candidate]))

        aggregate = {
            candidate: _median(view_scores[candidate]) for candidate in candidate_values
        }
        best = max(aggregate.values())
        selected = min(
            candidate for candidate, score in aggregate.items() if score == best
        )
        progress.set(selected_time=selected, selected_score=best)
        reporter.emit(
            "calibration_scores",
            candidate_scores=aggregate,
            selected_time=selected,
            selected_score=best,
            total_jvps=total_jvps,
        )
        return selected
