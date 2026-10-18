"""ADEval metrics used by the main tables."""

from __future__ import annotations

import numpy as np
import torch
from adeval import EvalAccumulator, EvalAccumulatorCuda

from .protocol import PROTOCOL


def _padded_bounds(lower: float, upper: float) -> tuple[float, float]:
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("metric inputs contain non-finite values")
    span = upper - lower
    padding = max(
        span * 1.0e-6,
        abs(lower) * np.finfo(np.float32).eps * 4,
        abs(upper) * np.finfo(np.float32).eps * 4,
        1.0e-7,
    )
    return lower - padding, upper + padding


def curve_area(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)[::-1]
    y = np.asarray(y, dtype=np.float64)[::-1]
    finite = np.isfinite(x) & np.isfinite(y)
    return float(np.trapezoid(y[finite], x[finite]))


def maximum_f1(curve_accumulator) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        recall, precision, _ = curve_accumulator.pr()
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(np.nanmax(f1))


def normalized_aupro(curve_accumulator) -> float:
    fpr, pro, _ = curve_accumulator.weighted_roc()
    fpr = np.asarray(fpr, dtype=np.float64)[::-1]
    pro = np.asarray(pro, dtype=np.float64)[::-1]
    finite = np.isfinite(fpr) & np.isfinite(pro)
    fpr, pro = fpr[finite], pro[finite]
    limit = PROTOCOL.aupro_fpr_limit
    below = fpr < limit
    clipped_fpr = np.concatenate((fpr[below], np.asarray([limit])))
    clipped_pro = np.concatenate(
        (pro[below], np.asarray([float(np.interp(limit, fpr, pro))]))
    )
    return float(np.trapezoid(clipped_pro, clipped_fpr) / limit)


def image_metrics(scores: list[float], labels: list[int]) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.uint8)
    bounds = _padded_bounds(float(values.min()), float(values.max()))
    accumulator = EvalAccumulator(
        bounds[0], bounds[1], skip_pixel_aupro=True, nstrips=PROTOCOL.metric_strips
    )
    accumulator.add_image(values, targets)
    curve = accumulator._image_acc
    with np.errstate(divide="ignore", invalid="ignore"):
        fpr, tpr, _ = curve.roc()
        recall, precision, _ = curve.pr()
    return {
        "auroc": curve_area(fpr, tpr),
        "aupr": curve_area(recall, precision),
        "f1": maximum_f1(curve),
    }


class PixelMetrics:
    def __init__(self, lower: float, upper: float):
        if not torch.cuda.is_available():
            raise RuntimeError("published ADEval evaluation requires CUDA")
        self.bounds = _padded_bounds(lower, upper)
        self.accumulator = EvalAccumulatorCuda(
            self.bounds[0],
            self.bounds[1],
            self.bounds[0],
            self.bounds[1],
            skip_pixel_aupro=False,
            nstrips=PROTOCOL.metric_strips,
        )

    def add(self, anomaly_map: np.ndarray, mask: np.ndarray) -> None:
        with torch.no_grad():
            self.accumulator.add_anomap(
                torch.as_tensor(anomaly_map, dtype=torch.float32, device="cuda"),
                torch.as_tensor(mask, dtype=torch.uint8, device="cuda"),
            )

    def result(self) -> dict[str, float]:
        curve = self.accumulator._pixel_acc
        with np.errstate(divide="ignore", invalid="ignore"):
            fpr, tpr, _ = curve.roc()
        return {
            "auroc": curve_area(fpr, tpr),
            "aupro": normalized_aupro(curve),
            "f1": maximum_f1(curve),
        }
