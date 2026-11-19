"""Phase-level progress, structured metrics, and reusable ETA measurements."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
import traceback
from uuid import uuid4


DEFAULT_LOG_INTERVAL = 30.0


def duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "estimating"
    hours, remainder = divmod(max(0, math.ceil(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def validation_epochs(maximum: int, frequency: int) -> list[int]:
    return sorted(set(range(frequency, maximum + 1, frequency)) | {maximum})


def training_eta(
    epoch: int,
    target: int,
    epoch_seconds: float | None,
    validation_seconds: float | None,
    schedule: list[int],
) -> float | None:
    if target <= epoch:
        return 0.0
    checks = sum(epoch < check <= target for check in schedule)
    if epoch_seconds is None or (checks and validation_seconds is None):
        return None
    return (target - epoch) * epoch_seconds + checks * (validation_seconds or 0.0)


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _value(value) -> str:
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _format_line(record: dict, verbose: bool = False) -> str:
    context = " ".join(
        str(record[key]) for key in ("command", "dataset") if key in record
    )
    if "shot" in record:
        context += f" {record['shot']}shot"
    if "seed" in record:
        context += f" s{record['seed']}"
    if "category" in record:
        context += f"/{record['category']}"
    prefix = f"{record['timestamp'][11:19]} {record['level']} [{context.strip()}]"
    if verbose:
        details = " ".join(
            f"{key}={_value(value)}"
            for key, value in record.items()
            if key not in {"timestamp", "run_id", "level", "traceback"}
            and value is not None
        )
        return f"{prefix} {details}"

    event = record["event"]
    phase = record.get("phase")
    if phase:
        suffix = (
            " started"
            if event == "stage_started"
            else " completed"
            if event == "stage_completed"
            else ""
        )
        parts = []
        if record["stage"] == "training":
            parts.append(
                f"epoch={record.get('epoch', record['completed'])}/{record.get('maximum_epochs', record['total'])}"
            )
        elif record["unit"] != "steps":
            parts.append(f"{record['completed']}/{record['total']} {record['unit']}")
        for key, label in (
            ("pass_name", "pass"),
            ("train_loss", "loss"),
            ("validation_loss", "val"),
            ("best_validation_loss", "best"),
            ("patience", "patience"),
            ("best_epoch", "best_epoch"),
            ("stop_reason", "stop"),
            ("selected_time", "t"),
            ("image_auroc", "image_AUROC"),
            ("pixel_auroc", "pixel_AUROC"),
        ):
            if record.get(key) is not None:
                parts.append(f"{label}={_value(record[key])}")
        parts.append(f"elapsed={duration(record['elapsed_s'])}")
        if event != "stage_completed":
            label = (
                "ETA(limit)"
                if record["stage"] == "training"
                else "ETA(pass)"
                if record["stage"] == "evaluation"
                and record.get("pass_name") != "final metrics"
                else "ETA"
            )
            parts.append(f"{label}={duration(record['eta_s'])}")
            if "early_stop_if_no_improvement_eta_s" in record:
                conditional = (
                    "epoch limit first"
                    if record.get("early_stop_status") == "epoch_limit_first"
                    else duration(record["early_stop_if_no_improvement_eta_s"])
                )
                parts.append(f"stop_if_unchanged={conditional}")
        return f"{prefix} {phase}{suffix} | " + " ".join(parts)

    parts = []
    if event == "evaluation_completed":
        for label, level, metric in (
            ("image_AUROC", "image", "auroc"),
            ("pixel_AUROC", "pixel", "auroc"),
            ("pixel_AUPRO", "pixel", "aupro"),
        ):
            value = record["metrics"][level][metric]
            parts.append(f"{label}={100 * value['mean']:.1f}±{100 * value['std']:.1f}%")
    for key in (
        "seeds",
        "categories",
        "output",
        "log_file",
        "metrics",
        "stage",
        "error",
        "nonfinite_loss",
        "nonfinite_gradient",
        "skipped_amp_updates",
    ):
        if record.get(key) is not None and not (
            event == "evaluation_completed" and key == "metrics"
        ):
            parts.append(f"{key}={_value(record[key])}")
    if event == "run_started":
        parts.append(f"device={record.get('profile', {}).get('device', 'unknown')}")
    if "elapsed_s" in record:
        parts.append(f"elapsed={duration(record['elapsed_s'])}")
    return f"{prefix} {event.replace('_', ' ')}" + (
        " | " + " ".join(parts) if parts else ""
    )


class RunLogger:
    """One reporting schedule for visible phases; full metrics stay in JSONL."""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        interval: float = DEFAULT_LOG_INTERVAL,
        verbose: bool = False,
        quiet: bool = False,
        context: dict | None = None,
        profile: dict | None = None,
        stream=None,
    ):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("log interval should be a positive finite number")
        self.interval, self.verbose, self.quiet = interval, verbose, quiet
        self.stream = sys.stderr if stream is None else stream
        self.context_fields = dict(context or {})
        self.profile = {"machine": platform.machine(), **(profile or {})}
        self._profile_key = hashlib.sha256(
            json.dumps(self.profile, sort_keys=True).encode()
        ).hexdigest()[:16]
        self._lock = threading.RLock()
        self._stages: list[Stage] = []
        self._history: dict = {}
        self._observations: dict = {}
        self._failure_context: dict | None = None
        self._last_report = float("-inf")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session = False
        self._text_file = self._json_file = None
        self._history_path: Path | None = None
        self.started = time.monotonic()
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        )
        if directory is not None:
            directory = Path(directory).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self.log_path = (
                directory
                / f"{self.context_fields.get('command', 'run')}-{self.run_id}.log"
            )
            self.jsonl_path = self.log_path.with_suffix(".jsonl")
            self._text_file = self.log_path.open("w", encoding="utf-8")
            self._json_file = self.jsonl_path.open("w", encoding="utf-8")
            self._history_path = directory / "timings.json"
            self._history = self._read_history()

    def _read_history(self) -> dict:
        if self._history_path is None:
            return {}
        try:
            value = json.loads(self._history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(value, dict):
            return {}
        # Cached estimates are optional; malformed entries should never fail a run.
        history = {}
        for key, entry in value.items():
            rate = entry.get("seconds_per_unit") if isinstance(entry, dict) else None
            if type(rate) in (int, float) and math.isfinite(rate) and rate > 0:
                history[key] = {"seconds_per_unit": rate}
        return history

    def seconds_per_unit(self, key: str, unit: str) -> float | None:
        entry = self._history.get(f"{self._profile_key}:{key}:{unit}", {})
        return entry.get("seconds_per_unit")

    def _remember(self, stage: Stage) -> None:
        if stage.completed <= 0:
            return
        observed = (time.monotonic() - stage.work_started) / stage.completed
        if observed <= 0:
            return
        for stage_key in {stage.key, stage.name}:
            key = f"{self._profile_key}:{stage_key}:{stage.unit}"
            previous = self.seconds_per_unit(stage_key, stage.unit)
            entry = {
                "seconds_per_unit": observed
                if previous is None
                else 0.7 * previous + 0.3 * observed
            }
            self._history[key] = self._observations[key] = entry

    def _save_history(self) -> None:
        if self._history_path is None:
            return
        temporary = None
        try:
            history = self._read_history()
            history.update(self._observations)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._history_path.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                json.dump(history, output, allow_nan=False, sort_keys=True)
            temporary.replace(self._history_path)
        except (OSError, ValueError) as error:
            self.emit("timing_cache_unavailable", level="WARNING", error=str(error))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def __enter__(self):
        self._session = True
        self._token = _ACTIVE.set(self)
        try:
            self.emit(
                "run_started",
                console=True,
                profile=self.profile,
                log_file=getattr(self, "log_path", None),
            )
        except BaseException:
            self._close()
            raise
        return self

    def _close(self) -> None:
        self._session = False
        _ACTIVE.reset(self._token)
        for output in (self._text_file, self._json_file):
            if output is not None:
                output.close()

    def __exit__(self, error_type, error, tb):
        self._session = False
        self._stop_heartbeat()
        try:
            if error is not None:
                self.emit(
                    "run_interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "run_failed",
                    level="ERROR",
                    error=str(error),
                    traceback="".join(
                        traceback.format_exception(error_type, error, tb)
                    ),
                    elapsed_s=time.monotonic() - self.started,
                    **(self._failure_context or {}),
                )
            else:
                self.emit(
                    "run_completed",
                    console=True,
                    elapsed_s=time.monotonic() - self.started,
                )
            self._save_history()
        finally:
            self._close()

    @contextmanager
    def context(self, **fields):
        with self._lock:
            previous = self.context_fields.copy()
            self.context_fields.update(fields)
        try:
            yield self
        finally:
            with self._lock:
                self.context_fields = previous

    def stage(self, name: str, total: int = 1, unit: str = "steps", **kwargs) -> Stage:
        return Stage(self, name, total, unit, **kwargs)

    def emit(
        self, event: str, *, level: str = "INFO", console: bool = False, **fields
    ) -> None:
        if level == "DEBUG" and not self.verbose:
            return
        with self._lock:
            record = _clean(
                {
                    "timestamp": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    "run_id": self.run_id,
                    "level": level,
                    **self.context_fields,
                    "event": event,
                    **fields,
                }
            )
            if self._stages:
                record.setdefault("stage", self._stages[-1].name)
            if self._json_file is not None:
                self._json_file.write(json.dumps(record, allow_nan=False) + "\n")
                self._json_file.flush()
            if not (console or self.verbose or level in {"WARNING", "ERROR"}):
                return
            line = _format_line(record, self.verbose)
            if self._text_file is not None:
                self._text_file.write(line + "\n")
                if record.get("traceback"):
                    self._text_file.write(record["traceback"])
                self._text_file.flush()
            if not self.quiet or level in {"WARNING", "ERROR"}:
                self.stream.write(line + "\n")
                self.stream.flush()
            self._last_report = time.monotonic()

    def _report_progress(self) -> None:
        with self._lock:
            if time.monotonic() - self._last_report < self.interval:
                return
            phase = next(
                (
                    stage
                    for stage in reversed(self._stages)
                    if stage.display is not None
                    and (not stage.transient or stage.elapsed >= self.interval)
                ),
                None,
            )
            if phase is not None:
                phase.shown = True
                self.emit("progress", console=True, **phase.snapshot())

    def _start_heartbeat(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = threading.Event()
        stop = self._stop

        def heartbeat():
            while not stop.wait(min(self.interval, 1.0)):
                self._report_progress()

        self._thread = threading.Thread(
            target=heartbeat, name="flownomaly-progress", daemon=True
        )
        self._thread.start()

    def _stop_heartbeat(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        self._thread = None


class Stage:
    """A timed operation; only stages with a display label are public phases."""

    def __init__(
        self,
        reporter: RunLogger,
        name: str,
        total: int,
        unit: str,
        *,
        key: str | None = None,
        display: str | None = None,
        transient: bool = False,
        **fields,
    ):
        self.reporter, self.name = reporter, name
        self.display, self.transient, self.shown = display, transient, False
        self.fields = fields
        self._countdowns: dict[str, float] = {}
        self.started = time.monotonic()
        self._finished = False
        self.set_work(total, unit, key=key or name)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float | None:
        if self._finished:
            return 0.0
        if self.completed >= self.total:
            children = self.reporter._stages
            return (
                children[-1].remaining
                if self in children and children[-1] is not self
                else None
            )
        if self._eta_override:
            estimate, updated = self._eta, self._eta_updated
        else:
            estimate = (
                None
                if self._rate is None
                else (self.total - self.completed) * self._rate
            )
            updated = self._last_advance
        remaining = (
            None if estimate is None else estimate - (time.monotonic() - updated)
        )
        return remaining if remaining is not None and remaining > 0 else None

    def set_work(self, total: int, unit: str, *, key: str | None = None) -> None:
        if total < 0:
            raise ValueError("stage total should be non-negative")
        with self.reporter._lock:
            if getattr(self, "completed", 0) > 0:
                self.reporter._remember(self)
            self.total, self.unit, self.completed = total, unit, 0
            self.key = key or self.name
            self.work_started = self._last_advance = time.monotonic()
            self._rate = self.reporter.seconds_per_unit(
                self.key, unit
            ) or self.reporter.seconds_per_unit(self.name, unit)
            self._samples = 0
            self._eta_override, self._eta = False, None
            self._eta_updated = self.work_started

    def snapshot(self) -> dict:
        fields = self.fields.copy()
        for key, updated in self._countdowns.items():
            value = fields[key]
            if value is not None:
                remaining = value - (time.monotonic() - updated)
                fields[key] = (
                    remaining if remaining > 0 else 0.0 if value == 0 else None
                )
        return {
            "stage": self.name,
            "phase": self.display,
            "completed": self.completed,
            "total": self.total,
            "unit": self.unit,
            "elapsed_s": self.elapsed,
            "eta_s": self.remaining,
            **fields,
        }

    def __enter__(self):
        with self.reporter._lock:
            self.started = self.work_started = self._last_advance = time.monotonic()
            self.reporter._stages.append(self)
            self.shown = self.display is not None and not self.transient
            self.reporter.emit(
                "stage_started",
                level="INFO" if self.display else "DEBUG",
                console=self.shown,
                **self.snapshot(),
            )
            if self.display is not None:
                self.reporter._start_heartbeat()
        return self

    def advance(self, count: int = 1, **fields) -> None:
        with self.reporter._lock:
            now = time.monotonic()
            if count > 0:
                observed = (now - self._last_advance) / count
                # Discard the first warm-up sample before smoothing later timings.
                self._rate = (
                    observed
                    if self._samples <= 1
                    else 0.7 * self._rate + 0.3 * observed
                )
                self._samples += 1
                self._last_advance = now
            self.completed += count
            self.set(**fields)

    def set(self, **fields) -> None:
        with self.reporter._lock:
            self.fields.update(fields)
            for key in fields:
                if key.endswith("_eta_s"):
                    self._countdowns[key] = time.monotonic()

    def eta(self, seconds: float | None) -> None:
        with self.reporter._lock:
            self._eta_override, self._eta = True, seconds
            self._eta_updated = time.monotonic()

    def __exit__(self, error_type, error, tb):
        with self.reporter._lock:
            if error is not None and self.reporter._failure_context is None:
                self.reporter._failure_context = {
                    **self.reporter.context_fields,
                    "stage": self.name,
                    "completed": self.completed,
                    "total": self.total,
                    "unit": self.unit,
                }
            if error is None:
                if self.completed < self.total:
                    self.advance(self.total - self.completed)
                self._finished = True
                self.reporter._remember(self)
            self.reporter.emit(
                "stage_completed" if error is None else "stage_failed",
                level="INFO" if self.display else "DEBUG",
                console=self.shown and error is None,
                **self.snapshot(),
                **({"error": str(error)} if error is not None else {}),
            )
            self.reporter._stages.remove(self)
            idle = not self.reporter._stages and not self.reporter._session
        if idle:
            self.reporter._stop_heartbeat()


_ACTIVE: ContextVar[RunLogger | None] = ContextVar("flownomaly_reporter", default=None)


def get_reporter() -> RunLogger:
    return _ACTIVE.get() or RunLogger()


def _positive_interval(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a positive number of seconds"
        ) from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite number of seconds")
    return number


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-interval",
        type=_positive_interval,
        default=DEFAULT_LOG_INTERVAL,
        help="Seconds between phase progress updates (default: 30).",
    )
    parser.add_argument("--log-dir", help="Log directory (default: <run output>/logs).")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include internal operations and diagnostic details.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print warnings/errors; keep logs on disk.",
    )


def logging_options(args) -> dict:
    return {
        key: getattr(args, key)
        for key in ("log_interval", "log_dir", "verbose", "quiet")
    }
