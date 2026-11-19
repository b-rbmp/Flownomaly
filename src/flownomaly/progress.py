"""Throttled progress, structured run logs, and reusable stage timings."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
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


def duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "estimating"
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def validation_epochs(maximum: int, frequency: int) -> list[int]:
    """Include the final epoch even when it falls outside the regular schedule."""
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
    remaining_checks = sum(epoch < check <= target for check in schedule)
    if epoch_seconds is None or (remaining_checks and validation_seconds is None):
        return None
    return (target - epoch) * epoch_seconds + remaining_checks * (
        validation_seconds or 0.0
    )


def _clean(value):
    """Keep JSON strict, including diagnostic events containing NaN or infinity."""
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
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_clean(value), separators=(",", ":"))
    return str(value)


def _console_line(now: datetime, record: dict) -> str:
    """Keep the terminal compact; the text/JSONL files retain every field."""
    context = " ".join(
        str(record[key]) for key in ("command", "dataset") if key in record
    )
    if "shot" in record:
        context += f" {record['shot']}shot"
    if "seed" in record:
        context += f" s{record['seed']}"
    if "category" in record:
        context += f"/{record['category']}"
    event = record["event"]
    stage = record.get("stage", "run")
    title = (
        stage
        if event == "progress"
        else (
            f"{stage} {'start' if event == 'stage_started' else 'done' if event == 'stage_completed' else 'failed'}"
            if event.startswith("stage_")
            else event
        )
    )
    parts = []
    if "completed" in record:
        parts.append(f"{record['completed']}/{record['total']} {record['unit']}")
    keys = {
        "run_started": ("log_file", "log_interval_s"),
        "run_plan": ("seeds", "categories", "fits", "workers"),
        "epoch_metrics": (
            "epoch",
            "train_loss",
            "learning_rate",
            "skipped_amp_updates",
        ),
        "validation_result": (
            "epoch",
            "train_loss",
            "validation_loss",
            "best_validation_loss",
            "best_epoch",
            "selected",
            "patience",
            "patience_limit",
            "decision",
        ),
        "training_completed": (
            "epochs",
            "final_train_loss",
            "best_epoch",
            "best_validation_loss",
            "stop_reason",
        ),
        "calibration_completed": (
            "selected_time",
            "selected_score",
            "runner_up",
            "score_gap",
            "boundary_selection",
        ),
        "calibration_view_completed": ("view", "total_views", "provisional_time"),
        "run_failed": ("stage", "error"),
        "run_interrupted": ("stage", "error"),
    }.get(
        event,
        (
            "epoch",
            "category_index",
            "seed_index",
            "batch",
            "train_loss",
            "validation_loss",
            "best_validation_loss",
            "patience",
            "next_validation_epoch",
            "view",
            "noise_pair",
            "candidate_time",
            "probe",
            "rotations_enabled",
            "retained_tokens",
            "valid_tokens",
            "stop_reason",
            "best_epoch",
            "operating_time",
            "training_shape",
            "validation_shape",
            "training_mib",
            "validation_mib",
            "free_gib",
            "path",
            "error",
            "skipped_amp_updates",
            "gradient_norm",
            "nonfinite_loss",
            "nonfinite_gradient",
            "metrics",
        ),
    )
    aliases = {
        "train_loss": "loss",
        "validation_loss": "val",
        "best_validation_loss": "best_val",
        "candidate_time": "t",
        "noise_pair": "pair",
        "next_validation_epoch": "next_val",
    }
    for key in keys:
        if event == "validation_result" and key in {"patience_limit", "selected"}:
            continue
        if record.get(key) is not None:
            value = duration(record[key]) if key.endswith("_s") else _value(record[key])
            if event == "validation_result" and key == "patience":
                value = f"{value}/{record['patience_limit']} checks"
            parts.append(f"{aliases.get(key, key)}={value}")
    if event == "stage_started" and stage == "training":
        parts.extend(
            f"{key}={_value(record[key])}"
            for key in ("validation_frequency", "patience_limit", "minimum_delta")
            if key in record
        )
    if event == "run_started":
        parts.append(f"device={record.get('profile', {}).get('device', 'unknown')}")
    if "elapsed_s" in record:
        parts.append(f"elapsed={duration(record['elapsed_s'])}")
    if "eta_s" in record:
        label = (
            "ETA(limit)"
            if record.get("eta_kind") == "epoch_limit" or stage == "training"
            else "ETA"
        )
        parts.append(f"{label}={duration(record['eta_s'])}")
        if record["eta_s"] is not None:
            parts.append(
                f"finish={(now + timedelta(seconds=record['eta_s'])).strftime('%H:%M:%S')}"
            )
    for label, key in (
        ("ETA(limit)", "limit_eta_s"),
        ("ETA(no improvement)", "early_stop_if_no_improvement_eta_s"),
        ("run ETA", "run_eta_s"),
    ):
        if key in record:
            value = (
                "epoch limit first"
                if key == "early_stop_if_no_improvement_eta_s"
                and record.get("early_stop_status") == "epoch_limit_first"
                else duration(record[key])
            )
            parts.append(f"{label}={value}")
    return (
        f"{now:%H:%M:%S} {record['level']} [{context.strip()}] {title} | "
        + " ".join(parts)
    )


class RunLogger:
    """One heartbeat per run; complete metrics on disk, compact output on stderr."""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        interval: float = 10.0,
        verbose: bool = False,
        quiet: bool = False,
        context: dict | None = None,
        profile: dict | None = None,
        stream=None,
    ):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("log interval should be a positive finite number")
        self.interval = interval
        self.verbose = verbose
        self.quiet = quiet
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
        self._timings: list[dict] = []
        self._failure_context: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session = False
        self._last_console = float("-inf")
        self._text_file = self._json_file = None
        self._history_path: Path | None = None
        self.started = time.monotonic()
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        )
        if directory is not None:
            directory = Path(directory).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            prefix = self.context_fields.get("command", "run")
            self.log_path = directory / f"{prefix}-{self.run_id}.log"
            self.jsonl_path = self.log_path.with_suffix(".jsonl")
            self._text_file = self.log_path.open("w", encoding="utf-8")
            self._json_file = self.jsonl_path.open("w", encoding="utf-8")
            self._history_path = directory / "timings.json"
            self._history = self._read_history()

    def _read_history(self) -> dict:
        try:
            value = json.loads(self._history_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, AttributeError):
            return {}

    def seconds_per_unit(self, key: str, unit: str) -> float | None:
        entry = self._history.get(f"{self._profile_key}:{key}:{unit}", {})
        value = entry.get("seconds_per_unit") if isinstance(entry, dict) else None
        return (
            value
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            else None
        )

    def _remember(self, stage: Stage) -> None:
        if stage.completed <= 0 or stage.elapsed <= 0:
            return
        observed = stage.elapsed / stage.completed
        for stage_key in {stage.key, stage.name}:
            key = f"{self._profile_key}:{stage_key}:{stage.unit}"
            previous = self.seconds_per_unit(stage_key, stage.unit)
            entry = {
                "seconds_per_unit": observed
                if previous is None
                else 0.7 * previous + 0.3 * observed
            }
            self._history[key] = self._observations[key] = entry

    def phase_times(self, parent_id: str) -> dict[str, float]:
        with self._lock:
            return {
                item["stage"]: round(item["elapsed_s"], 3)
                for item in self._timings
                if item["parent_id"] == parent_id
            }

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
        except OSError as error:
            self.emit("timing_cache_unavailable", level="WARNING", error=str(error))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def __enter__(self):
        self._session = True
        self._token = _ACTIVE.set(self)
        self.emit(
            "run_started",
            profile=self.profile,
            log_file=getattr(self, "log_path", None),
            metrics_file=getattr(self, "jsonl_path", None),
            log_interval_s=self.interval,
        )
        return self

    def __exit__(self, error_type, error, tb):
        self._session = False
        self._stop_heartbeat()
        if error is not None:
            self.emit(
                "run_interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "run_failed",
                level="ERROR",
                error=str(error),
                traceback="".join(traceback.format_exception(error_type, error, tb)),
                elapsed_s=time.monotonic() - self.started,
                **(self._failure_context or {}),
            )
        else:
            self.emit(
                "run_completed", elapsed_s=time.monotonic() - self.started, eta_s=0.0
            )
        self._save_history()
        _ACTIVE.reset(self._token)
        for output in (self._text_file, self._json_file):
            if output is not None:
                output.close()

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

    def emit(self, event: str, *, level: str = "INFO", console=True, **fields) -> None:
        with self._lock:
            now = datetime.now().astimezone()
            record = _clean(
                {
                    "timestamp": now.isoformat(timespec="seconds"),
                    "run_id": self.run_id,
                    "level": level,
                    **self.context_fields,
                    "event": event,
                    **fields,
                }
            )
            if self._stages:
                record.setdefault("stage", self._stages[-1].name)
            labels = " ".join(
                f"{key}={record[key]}"
                for key in ("command", "dataset", "shot", "seed", "category")
                if key in record
            )
            parts = []
            for key, value in fields.items():
                if key in {"traceback", "profile", "stage_id"}:
                    continue
                if (
                    key.endswith("_s")
                    and not key.endswith("_per_s")
                    and (value is None or isinstance(value, (int, float)))
                ):
                    parts.append(f"{key[:-2]}={duration(value)}")
                elif value is not None:
                    parts.append(f"{key}={_value(value)}")
            if "eta_s" in fields and fields["eta_s"] is not None:
                parts.append(
                    f"finish={(now + timedelta(seconds=max(0, fields['eta_s']))).strftime('%H:%M:%S')}"
                )
            line = f"{now:%H:%M:%S} {level} [{labels}] {event}"
            if parts:
                line += " | " + " ".join(parts)
            if self._text_file is not None:
                self._text_file.write(line + "\n")
                if fields.get("traceback"):
                    self._text_file.write(fields["traceback"])
                self._text_file.flush()
                self._json_file.write(json.dumps(record, allow_nan=False) + "\n")
                self._json_file.flush()
            due = time.monotonic() - self._last_console >= self.interval
            visible = (
                level in {"WARNING", "ERROR"}
                or (not self.quiet and (level != "DEBUG" or self.verbose))
            ) and (console is True or (console == "periodic" and due))
            if visible:
                self.stream.write(
                    (line if self.verbose else _console_line(now, record)) + "\n"
                )
                self.stream.flush()
                self._last_console = time.monotonic()

    def _start_heartbeat(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = threading.Event()
            stop = self._stop

            def heartbeat():
                while not stop.wait(self.interval):
                    with self._lock:
                        if not self._stages:
                            continue
                        active = self._stages[-1]
                        fields = active.snapshot()
                        for parent in self._stages[:-1]:
                            if parent.name in {"run", "seed", "category", "training"}:
                                fields[f"{parent.name}_eta_s"] = parent.remaining
                                fields = {**parent.detail_fields(), **fields}
                        self.emit("progress", console="periodic", **fields)

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
    def __init__(
        self,
        reporter: RunLogger,
        name: str,
        total: int,
        unit: str,
        *,
        key: str | None = None,
        quiet: bool = False,
        **fields,
    ):
        if total < 0:
            raise ValueError("stage total should be non-negative")
        self.reporter, self.name, self.total, self.unit = reporter, name, total, unit
        self.key, self.quiet = key or name, quiet
        self.fields = fields
        self._field_updated: dict[str, float] = {}
        self._parent_id: str | None = None
        self.completed = 0
        self.started = self._last_advance = time.monotonic()
        self._rate = reporter.seconds_per_unit(
            self.key, unit
        ) or reporter.seconds_per_unit(name, unit)
        self._samples = 0
        self._eta_override = False
        self._eta = None
        self._eta_updated = self.started
        self.stage_id = uuid4().hex[:8]

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float | None:
        if self.completed >= self.total:
            if (
                self.reporter._stages
                and self in self.reporter._stages
                and self.reporter._stages[-1] is not self
            ):
                return self.reporter._stages[-1].remaining
            return 0.0
        if self._eta_override:
            estimate = self._eta
            since = time.monotonic() - self._eta_updated
        else:
            estimate = (
                None
                if self._rate is None
                else (self.total - self.completed) * self._rate
            )
            since = time.monotonic() - self._last_advance
        # An overrun is unknown, rather than a misleading zero before completion.
        return estimate - since if estimate is not None and estimate > since else None

    def snapshot(self) -> dict:
        remaining = self.remaining
        return {
            "stage": self.name,
            "stage_id": self.stage_id,
            "completed": self.completed,
            "total": self.total,
            "unit": self.unit,
            "elapsed_s": self.elapsed,
            "eta_s": remaining,
            "rate_per_s": None if not self._rate else 1.0 / self._rate,
            "eta_source": "projection"
            if self._eta_override
            else "measured"
            if self._samples
            else "history"
            if self._rate
            else "estimating",
            **self.detail_fields(),
        }

    def detail_fields(self) -> dict:
        fields = self.fields.copy()
        for key, updated in self._field_updated.items():
            if fields[key] is not None:
                remaining = fields[key] - (time.monotonic() - updated)
                fields[key] = (
                    remaining if remaining > 0 else 0.0 if fields[key] == 0 else None
                )
        return fields

    def __enter__(self):
        with self.reporter._lock:
            self.started = self._last_advance = time.monotonic()
            self._parent_id = (
                self.reporter._stages[-1].stage_id if self.reporter._stages else None
            )
            self.reporter._stages.append(self)
            self.reporter.emit(
                "stage_started", console=not self.quiet, **self.snapshot()
            )
            self.reporter._start_heartbeat()
        return self

    def advance(self, count: int = 1, **fields) -> None:
        with self.reporter._lock:
            now = time.monotonic()
            if count > 0:
                observed = (now - self._last_advance) / count
                # Replace the first (warm-up) sample on the second update.
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
                if key.endswith("eta_s"):
                    self._field_updated[key] = time.monotonic()

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
                self.reporter._remember(self)
                self.reporter._timings.append(
                    {
                        "stage": self.name,
                        "parent_id": self._parent_id,
                        "elapsed_s": self.elapsed,
                    }
                )
            self.reporter.emit(
                "stage_completed" if error is None else "stage_failed",
                console=not self.quiet if error is None else False,
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
        default=10.0,
        help="Seconds between progress/metric updates (default: 10).",
    )
    parser.add_argument("--log-dir", help="Log directory (default: <run output>/logs).")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed diagnostic events."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print warnings/errors; keep full logs on disk.",
    )


def logging_options(args) -> dict:
    return {
        key: getattr(args, key)
        for key in ("log_interval", "log_dir", "verbose", "quiet")
    }
