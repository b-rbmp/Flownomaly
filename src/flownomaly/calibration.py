"""Normal-only Jacobian gain calibration for the operating time."""

from __future__ import annotations

import hashlib
import math

import torch

from .protocol import PROTOCOL
from .progress import RunLogger, get_reporter
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
    with reporter.stage(
        "calibration_search",
        total_jvps,
        "JVPs",
        views=len(calibration_views),
        candidates=len(candidate_values),
        noise_pairs=PROTOCOL.jgn_noise_pairs,
        probes=PROTOCOL.jgn_probes,
    ) as progress:
        for view_index, view in enumerate(calibration_views, 1):
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
                            progress.advance(
                                view=f"{view_index}/{len(calibration_views)}",
                                view_id=view["view_id"],
                                noise_pair=f"{pair_index + 1}/{PROTOCOL.jgn_noise_pairs}",
                                candidate_time=candidate,
                                chunk=start // PROTOCOL.jgn_batch_size + 1,
                                probe=f"{probe_index + 1}/{PROTOCOL.jgn_probes}",
                            )
                    gains = [value / token_count for value in probe_sums]
                    gain = sum(gains) / PROTOCOL.jgn_probes
                    scores = torch.cat(residual_scores)
                    normal_floor = float(
                        torch.topk(scores, top_count, largest=True, sorted=False)
                        .values.mean()
                        .item()
                    )
                    ratio = gain / (normal_floor + PROTOCOL.jgn_epsilon)
                    pair_ratios[candidate].append(ratio)
                    reporter.emit(
                        "calibration_candidate",
                        level="DEBUG",
                        console=reporter.verbose,
                        view_id=view["view_id"],
                        noise_pair=pair_index + 1,
                        candidate_time=candidate,
                        gain=gain,
                        normal_floor=normal_floor,
                        score=ratio,
                        valid_tokens=token_count,
                        top_tokens=top_count,
                    )
                    if not math.isfinite(ratio):
                        reporter.emit(
                            "nonfinite_calibration_score",
                            level="WARNING",
                            console="periodic",
                            view_id=view["view_id"],
                            candidate_time=candidate,
                            gain=gain,
                            normal_floor=normal_floor,
                            score=ratio,
                        )

            for candidate in candidate_values:
                view_scores[candidate].append(_median(pair_ratios[candidate]))
            provisional = {
                candidate: _median(scores) for candidate, scores in view_scores.items()
            }
            leader = max(
                provisional, key=lambda candidate: (provisional[candidate], -candidate)
            )
            progress.set(
                provisional_time=leader,
                provisional_score=provisional[leader],
                completed_views=view_index,
                scores_status="provisional",
            )
            reporter.emit(
                "calibration_view_completed",
                console="periodic",
                view=view_index,
                total_views=len(calibration_views),
                provisional_time=leader,
                provisional_scores=provisional,
                scores_status="provisional",
                eta_s=progress.remaining,
            )

    aggregate = {
        candidate: _median(view_scores[candidate]) for candidate in candidate_values
    }
    best = max(aggregate.values())
    selected = min(candidate for candidate, score in aggregate.items() if score == best)
    ranked = sorted(aggregate, key=lambda candidate: (-aggregate[candidate], candidate))
    runner_up = ranked[1] if len(ranked) > 1 else None
    reporter.emit(
        "calibration_completed",
        selected_time=selected,
        selected_score=best,
        runner_up=runner_up,
        score_gap=None if runner_up is None else best - aggregate[runner_up],
        boundary_selection=selected in (min(candidate_values), max(candidate_values)),
        candidate_scores=aggregate,
        total_jvps=total_jvps,
        elapsed_s=progress.elapsed,
        eta_s=0.0,
    )
    return selected
