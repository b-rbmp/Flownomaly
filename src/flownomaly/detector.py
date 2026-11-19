"""Flownomaly fitting and inference path."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TF

from .calibration import select_operating_time
from .data import Sample
from .network import VelocityNetwork
from .protocol import PROTOCOL
from .progress import RunLogger, Stage, get_reporter, training_eta, validation_epochs
from .rotation import rotation_view


class Flownomaly:
    def __init__(self, device: str = "cuda", *, reporter: RunLogger | None = None):
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The reproduction path requires a CUDA GPU.")
        self.device = torch.device(device)
        self.reporter = reporter if reporter is not None else get_reporter()
        self.fit_summary: dict = {}
        self.encoder: nn.Module | None = None
        self.velocity_model: VelocityNetwork | None = None
        self.layer_statistics: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.operating_time: float | None = None
        self._hook_outputs: dict[int, torch.Tensor] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (PROTOCOL.image_size, PROTOCOL.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _hook(self, layer: int):
        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, (list, tuple)) else output
            self._hook_outputs[layer] = value.detach().cpu()

        return capture

    def load_encoder(self) -> None:
        if self.encoder is not None:
            return
        with self.reporter.stage(
            "encoder_loading",
            display="Loading encoder",
            transient=True,
            backbone=PROTOCOL.backbone,
        ):
            self._load_encoder()

    def _load_encoder(self) -> None:
        encoder = (
            torch.hub.load(
                "facebookresearch/dinov2:main",
                PROTOCOL.backbone,
                source="github",
                trust_repo=True,
            )
            .to(self.device)
            .half()
        )
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        if len(encoder.blocks) <= max(PROTOCOL.target_layers):
            raise RuntimeError("The loaded DINOv2 encoder has an unexpected depth.")
        if int(encoder.patch_size) != PROTOCOL.patch_size:
            raise RuntimeError(
                "The loaded DINOv2 encoder has an unexpected patch size."
            )
        for layer in PROTOCOL.target_layers:
            encoder.blocks[layer].register_forward_hook(self._hook(layer))
        self.encoder = encoder

    @torch.no_grad()
    def _extract_raw(self, images: torch.Tensor) -> list[torch.Tensor]:
        if self.encoder is None:
            raise RuntimeError("load_encoder() must be called first")
        self._hook_outputs = {}
        images = images.to(self.device, non_blocking=True)
        encoder_dtype = next(self.encoder.parameters()).dtype
        with torch.amp.autocast(device_type="cuda", enabled=True):
            self.encoder(images.to(encoder_dtype))
        features = []
        expected_tokens = PROTOCOL.grid_size**2
        for layer in PROTOCOL.target_layers:
            value = self._hook_outputs[layer]
            if value.shape[1] > expected_tokens:
                value = value[:, -expected_tokens:, :]
            if value.shape[1] != expected_tokens or value.shape[-1] != 1536:
                raise RuntimeError(
                    f"unexpected DINOv2 feature shape at layer {layer}: {tuple(value.shape)}"
                )
            features.append(value.contiguous().float())
        return features

    def _normalize_and_average(self, features: list[torch.Tensor]) -> torch.Tensor:
        normalized = []
        for layer, value in zip(PROTOCOL.target_layers, features, strict=True):
            mean, standard_deviation = self.layer_statistics[layer]
            normalized.append(
                (value - mean.to(value.device))
                / (standard_deviation.to(value.device) + 1.0e-6)
            )
        return torch.stack(normalized).mean(dim=0)

    @staticmethod
    def _valid_token_mask(angle_degrees: float) -> torch.Tensor:
        grid = PROTOCOL.grid_size
        coordinates = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid - 0.5
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        angle = math.radians(angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)
        source_x = cosine * xx - sine * yy
        source_y = sine * xx + cosine * yy
        return ((source_x.abs() <= 0.5) & (source_y.abs() <= 0.5)).reshape(-1)

    @staticmethod
    def _rotations_enabled(dataset: str, category: str) -> bool:
        return not (dataset == "mvtec" and "transistor" in category.lower())

    def _rotation_pool(
        self,
        support_images: torch.Tensor,
        support_samples: list[Sample],
        namespace: str,
        count: int,
        seed: int,
        offset: int,
        enabled: bool,
        progress: Stage,
    ) -> dict[int, list[torch.Tensor]]:
        per_layer = {layer: [] for layer in PROTOCOL.target_layers}
        maximum_angle = PROTOCOL.maximum_rotation_angle if enabled else 0.0
        for augmentation_index in range(count):
            for start in range(0, len(support_samples), PROTOCOL.extraction_batch_size):
                stop = min(start + PROTOCOL.extraction_batch_size, len(support_samples))
                views = [
                    rotation_view(
                        support_samples[index].image_key,
                        namespace,
                        augmentation_index,
                        seed,
                        offset,
                        maximum_angle,
                    )
                    for index in range(start, stop)
                ]
                rotated = torch.stack(
                    [
                        support_images[index]
                        if view.angle_degrees == 0.0
                        else TF.rotate(
                            support_images[index],
                            view.angle_degrees,
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            fill=0.0,
                        )
                        for index, view in zip(range(start, stop), views, strict=True)
                    ]
                )
                raw = self._extract_raw(rotated)
                masks = torch.stack(
                    [self._valid_token_mask(view.angle_degrees) for view in views]
                )
                for layer, value in zip(PROTOCOL.target_layers, raw, strict=True):
                    per_layer[layer].append(value[masks.to(value.device)])
                progress.advance(stop - start)
        return per_layer

    def _identity_pool(
        self, support_images: torch.Tensor, progress: Stage
    ) -> dict[int, list[torch.Tensor]]:
        per_layer = {layer: [] for layer in PROTOCOL.target_layers}
        for start in range(0, len(support_images), PROTOCOL.extraction_batch_size):
            raw = self._extract_raw(
                support_images[start : start + PROTOCOL.extraction_batch_size]
            )
            for layer, value in zip(PROTOCOL.target_layers, raw, strict=True):
                per_layer[layer].append(value.reshape(-1, value.shape[-1]))
            progress.advance(len(raw[0]))
        return per_layer

    def _fixed_validation_loss(
        self, validation_bank: torch.Tensor, *, epoch: int | None = None
    ) -> float:
        if self.velocity_model is None:
            raise RuntimeError("velocity model is not initialized")
        was_training = self.velocity_model.training
        self.velocity_model.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(PROTOCOL.validation_seed)
        total = 0.0
        count = 0
        try:
            with (
                self.reporter.stage(
                    "validation",
                    len(validation_bank),
                    "tokens",
                    display="Validation",
                    transient=True,
                    epoch=epoch,
                    key=f"validation:{len(validation_bank)}",
                    batches=math.ceil(len(validation_bank) / PROTOCOL.flow_batch_size),
                ) as progress,
                torch.no_grad(),
            ):
                for start in range(0, len(validation_bank), PROTOCOL.flow_batch_size):
                    target = validation_bank[
                        start : start + PROTOCOL.flow_batch_size
                    ].float()
                    noise = torch.randn(
                        target.shape, generator=generator, dtype=torch.float32
                    ).to(self.device)
                    time = torch.rand(
                        (len(target), 1), generator=generator, dtype=torch.float32
                    ).to(self.device)
                    state = time * target + (1.0 - time) * noise
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        prediction = self.velocity_model(state, time)
                    loss = torch.nn.functional.mse_loss(
                        prediction.float(), (target - noise).float()
                    )
                    total += float(loss.item()) * len(target)
                    count += len(target)
                    progress.advance(len(target), validation_loss=total / count)
            return total / count
        finally:
            self.velocity_model.train(was_training)

    def fit(
        self,
        support_images: torch.Tensor,
        support_samples: list[Sample],
        dataset: str,
        category: str,
    ) -> None:
        with self.reporter.context(dataset=dataset, category=category):
            self._fit(support_images, support_samples, dataset, category)

    def _feature_banks(
        self,
        support_images: torch.Tensor,
        support_samples: list[Sample],
        rotations_enabled: bool,
        progress: Stage,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        training = self._identity_pool(support_images, progress)
        if rotations_enabled:
            rotated_training = self._rotation_pool(
                support_images,
                support_samples,
                "train",
                PROTOCOL.training_rotations,
                PROTOCOL.training_rotation_seed,
                0,
                True,
                progress,
            )
            for layer in PROTOCOL.target_layers:
                training[layer].extend(rotated_training[layer])
            del rotated_training
        validation = self._rotation_pool(
            support_images,
            support_samples,
            "cfm_val",
            PROTOCOL.validation_rotations,
            PROTOCOL.validation_seed,
            PROTOCOL.validation_rotation_offset,
            rotations_enabled,
            progress,
        )
        with self.reporter.stage(
            "feature_statistics", len(PROTOCOL.target_layers), "layers"
        ) as statistics:
            for layer in PROTOCOL.target_layers:
                values = torch.cat(training[layer], dim=0)
                self.layer_statistics[layer] = (
                    values.mean(dim=0, keepdim=True),
                    values.std(dim=0, keepdim=True),
                )
                del values
                statistics.advance()
        with self.reporter.stage("normalize_training_bank"):
            training_bank = self._normalize_and_average(
                [torch.cat(training[layer], dim=0) for layer in PROTOCOL.target_layers]
            )
        with self.reporter.stage("normalize_validation_bank"):
            validation_bank = self._normalize_and_average(
                [
                    torch.cat(validation[layer], dim=0)
                    for layer in PROTOCOL.target_layers
                ]
            )
        self.reporter.emit(
            "feature_banks",
            level="DEBUG",
            training_shape=list(training_bank.shape),
            validation_shape=list(validation_bank.shape),
            rotations_enabled=rotations_enabled,
        )
        return training_bank, validation_bank

    def _fit(
        self,
        support_images: torch.Tensor,
        support_samples: list[Sample],
        dataset: str,
        category: str,
    ) -> None:
        if self.encoder is None:
            raise RuntimeError("load_encoder() must be called first")
        self.operating_time = None
        self.fit_summary = {}
        rotations_enabled = self._rotations_enabled(dataset, category)
        view_count = len(support_images) * (
            1
            + (PROTOCOL.training_rotations if rotations_enabled else 0)
            + PROTOCOL.validation_rotations
        )
        with self.reporter.stage(
            "feature_extraction", view_count, "views", display="Extracting features"
        ) as progress:
            training_bank, validation_bank = self._feature_banks(
                support_images, support_samples, rotations_enabled, progress
            )

        with self.reporter.stage("velocity_initialization"):
            self.velocity_model = VelocityNetwork().to(self.device)
        encoder_was_cuda = next(self.encoder.parameters()).device.type == "cuda"
        if encoder_was_cuda:
            with self.reporter.stage("encoder_offload"):
                self.encoder.to("cpu")
                torch.cuda.empty_cache()
        try:
            try:
                with self.reporter.stage("feature_bank_transfer"):
                    training_bank = training_bank.to(self.device)
                    validation_bank = validation_bank.to(self.device)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                        if self.reporter.verbose:
                            free, total = torch.cuda.mem_get_info(self.device)
                            self.reporter.emit(
                                "gpu_memory",
                                level="DEBUG",
                                free_gib=free / 2**30,
                                total_gib=total / 2**30,
                            )
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    "The training and validation feature banks do not fit in GPU memory."
                ) from error
            self._train_velocity(training_bank, validation_bank)
        finally:
            del training_bank, validation_bank
            torch.cuda.empty_cache()
            if encoder_was_cuda:
                with self.reporter.stage("encoder_restore"):
                    self.encoder.to(self.device)
                    torch.cuda.synchronize(self.device)

        with self.reporter.stage(
            "calibration",
            PROTOCOL.calibration_rotations * len(support_samples),
            "views",
            key="calibration_features",
            display="Calibration",
        ) as progress:
            progress.eta(None)
            calibration_views = self._calibration_views(
                support_images, support_samples, rotations_enabled, progress
            )
            self.operating_time = select_operating_time(
                self.velocity_model,
                category,
                calibration_views,
                self.device,
                reporter=self.reporter,
                progress=progress,
            )
        self.fit_summary["operating_time"] = self.operating_time

    def _train_velocity(
        self, training_bank: torch.Tensor, validation_bank: torch.Tensor
    ) -> None:
        optimizer = torch.optim.AdamW(
            self.velocity_model.parameters(),
            lr=PROTOCOL.learning_rate,
            weight_decay=PROTOCOL.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=PROTOCOL.maximum_epochs,
            eta_min=PROTOCOL.minimum_learning_rate,
        )
        permutation_generator = torch.Generator(device=self.device)
        permutation_generator.manual_seed(PROTOCOL.training_rotation_seed)
        scaler = torch.amp.GradScaler("cuda")
        best_loss = float("inf")
        best_weights = None
        best_epoch = None
        patience = 0
        schedule = validation_epochs(
            PROTOCOL.maximum_epochs, PROTOCOL.validation_frequency
        )
        epoch_key = f"training_epoch:{len(training_bank)}"
        epoch_rate = self.reporter.seconds_per_unit(
            epoch_key, "tokens"
        ) or self.reporter.seconds_per_unit("training_epoch", "tokens")
        epoch_seconds = None if epoch_rate is None else epoch_rate * len(training_bank)
        val_rate = self.reporter.seconds_per_unit(
            f"validation:{len(validation_bank)}", "tokens"
        ) or self.reporter.seconds_per_unit("validation", "tokens")
        validation_seconds = (
            None if val_rate is None else val_rate * len(validation_bank)
        )
        stop_reason = "maximum_epochs"
        optimizer_steps = 0

        def count_optimizer_step(_optimizer, _args, _kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1

        # AMP can skip an optimizer step; count actual updates without reading GPU scalars.
        step_hook = optimizer.register_step_post_hook(count_optimizer_step)
        try:
            with self.reporter.stage(
                "training",
                PROTOCOL.maximum_epochs,
                "epochs",
                display="Training",
                key=f"training:{len(training_bank)}",
                maximum_epochs=PROTOCOL.maximum_epochs,
            ) as progress:
                progress.eta(
                    training_eta(
                        0,
                        PROTOCOL.maximum_epochs,
                        epoch_seconds,
                        validation_seconds,
                        schedule,
                    )
                )
                for epoch in range(1, PROTOCOL.maximum_epochs + 1):
                    progress.set(epoch=epoch)
                    epoch_started = time.monotonic()
                    self.velocity_model.train()
                    permutation = torch.randperm(
                        len(training_bank),
                        generator=permutation_generator,
                        device=self.device,
                    )
                    loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
                    tokens_seen = 0
                    steps_before = optimizer_steps
                    learning_rate = optimizer.param_groups[0]["lr"]
                    detail_updated = time.monotonic()
                    with self.reporter.stage(
                        "training_epoch",
                        len(training_bank),
                        "tokens",
                        key=epoch_key,
                        epoch=epoch,
                    ) as epoch_progress:
                        for start in range(
                            0, len(training_bank), PROTOCOL.flow_batch_size
                        ):
                            target = training_bank.index_select(
                                0, permutation[start : start + PROTOCOL.flow_batch_size]
                            ).float()
                            noise = torch.randn_like(target)
                            times = torch.rand((len(target), 1), device=self.device)
                            state = times * target + (1.0 - times) * noise
                            optimizer.zero_grad()
                            with torch.amp.autocast(device_type="cuda", enabled=True):
                                prediction = self.velocity_model(state, times)
                                loss = torch.nn.functional.mse_loss(
                                    prediction.float(), (target - noise).float()
                                )
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            gradient_norm = torch.nn.utils.clip_grad_norm_(
                                self.velocity_model.parameters(), max_norm=1.0
                            )
                            scaler.step(optimizer)
                            scaler.update()
                            loss_sum.add_(loss.detach().double() * len(target))
                            tokens_seen += len(target)
                            epoch_progress.advance(len(target))
                            if (
                                time.monotonic() - detail_updated
                                >= self.reporter.interval
                            ):
                                progress.set(
                                    train_loss=float(loss_sum.item()) / tokens_seen
                                )
                                detail_updated = time.monotonic()
                        epoch_loss = float(loss_sum.item()) / tokens_seen
                        norm = float(gradient_norm.item())
                    scheduler.step()
                    epoch_duration = time.monotonic() - epoch_started
                    epoch_seconds = (
                        epoch_duration
                        if epoch <= 2
                        else 0.7 * epoch_seconds + 0.3 * epoch_duration
                    )
                    skipped = math.ceil(
                        len(training_bank) / PROTOCOL.flow_batch_size
                    ) - (optimizer_steps - steps_before)
                    if (
                        not math.isfinite(epoch_loss)
                        or not math.isfinite(norm)
                        or skipped
                    ):
                        self.reporter.emit(
                            "training_numerics",
                            level="WARNING",
                            epoch=epoch,
                            nonfinite_loss=not math.isfinite(epoch_loss),
                            nonfinite_gradient=not math.isfinite(norm),
                            skipped_amp_updates=skipped,
                        )
                    self.reporter.emit(
                        "epoch_metrics",
                        epoch=epoch,
                        train_loss=epoch_loss,
                        gradient_norm=norm,
                        learning_rate=learning_rate,
                        next_learning_rate=optimizer.param_groups[0]["lr"],
                        epoch_elapsed_s=epoch_duration,
                        tokens_per_s=tokens_seen / epoch_duration,
                        skipped_amp_updates=skipped,
                        amp_scale=scaler.get_scale(),
                    )
                    if epoch in schedule:
                        validation_started = time.monotonic()
                        validation_loss = self._fixed_validation_loss(
                            validation_bank, epoch=epoch
                        )
                        improvement = (
                            best_loss - validation_loss
                            if math.isfinite(best_loss)
                            else None
                        )
                        improved = (
                            validation_loss
                            < best_loss - PROTOCOL.validation_minimum_delta
                        )
                        if improved:
                            best_loss, best_epoch = validation_loss, epoch
                            best_weights = copy.deepcopy(
                                self.velocity_model.state_dict()
                            )
                            patience = 0
                        else:
                            patience += 1
                        measured = time.monotonic() - validation_started
                        validation_seconds = (
                            measured
                            if validation_seconds is None
                            else 0.7 * validation_seconds + 0.3 * measured
                        )
                        if not math.isfinite(validation_loss):
                            self.reporter.emit(
                                "nonfinite_validation_loss",
                                level="WARNING",
                                epoch=epoch,
                                error="validation loss is non-finite",
                            )
                        self.reporter.emit(
                            "validation_result",
                            epoch=epoch,
                            train_loss=epoch_loss,
                            validation_loss=validation_loss,
                            best_validation_loss=best_loss,
                            best_epoch=best_epoch,
                            improvement=improvement,
                            minimum_delta=PROTOCOL.validation_minimum_delta,
                            selected=improved,
                            patience=patience,
                            patience_limit=PROTOCOL.early_stopping_patience,
                            patience_unit="validation_checks",
                            validation_elapsed_s=measured,
                        )
                        progress.set(validation_loss=validation_loss)
                        if patience >= PROTOCOL.early_stopping_patience:
                            stop_reason = "early_stopping"
                    progress.advance()
                    future = [check for check in schedule if check > epoch]
                    checks_left = PROTOCOL.early_stopping_patience - patience
                    if checks_left == 0:
                        stop_epoch = epoch
                    elif len(future) >= checks_left:
                        stop_epoch = future[checks_left - 1]
                    else:
                        stop_epoch = None
                    progress.eta(
                        training_eta(
                            epoch,
                            PROTOCOL.maximum_epochs,
                            epoch_seconds,
                            validation_seconds,
                            schedule,
                        )
                    )
                    progress.set(
                        train_loss=epoch_loss,
                        best_validation_loss=best_loss,
                        best_epoch=best_epoch,
                        patience=f"{patience}/{PROTOCOL.early_stopping_patience} checks",
                        early_stop_status="epoch_limit_first"
                        if stop_epoch is None
                        else "conditional",
                        early_stop_if_no_improvement_eta_s=None
                        if stop_epoch is None
                        else training_eta(
                            epoch,
                            stop_epoch,
                            epoch_seconds,
                            validation_seconds,
                            schedule,
                        ),
                    )
                    if stop_reason == "early_stopping":
                        break
                if best_weights is None:
                    raise RuntimeError("no validation checkpoint was selected")
                with self.reporter.stage("restore_best_weights"):
                    self.velocity_model.load_state_dict(best_weights)
                    self.velocity_model.eval()
                    for parameter in self.velocity_model.parameters():
                        parameter.grad = None
                        parameter.requires_grad_(False)
                self.fit_summary = {
                    "epochs": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "final_train_loss": epoch_loss,
                    "stop_reason": stop_reason,
                }
                progress.total = epoch
                progress.set(stop_reason=stop_reason)
        finally:
            step_hook.remove()

    def _calibration_views(
        self,
        support_images: torch.Tensor,
        support_samples: list[Sample],
        rotations_enabled: bool,
        progress: Stage,
    ) -> list[dict]:
        maximum_angle = PROTOCOL.maximum_rotation_angle if rotations_enabled else 0.0
        views = []
        for augmentation_index in range(PROTOCOL.calibration_rotations):
            for image_index, (image, sample) in enumerate(
                zip(support_images, support_samples, strict=True)
            ):
                view = rotation_view(
                    sample.image_key,
                    "calibration",
                    augmentation_index,
                    PROTOCOL.calibration_seed,
                    PROTOCOL.calibration_rotation_offset,
                    maximum_angle,
                )
                transformed = (
                    image
                    if view.angle_degrees == 0.0
                    else TF.rotate(
                        image,
                        view.angle_degrees,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        fill=0.0,
                    )
                )
                raw = [
                    value.half().float()
                    for value in self._extract_raw(transformed[None])
                ]
                features = self._normalize_and_average(raw).squeeze(0)
                features = features[
                    self._valid_token_mask(view.angle_degrees).to(features.device)
                ]
                views.append(
                    {
                        "view_id": f"support_{image_index}_calibration_{augmentation_index}",
                        "image_key": sample.image_key,
                        "augmentation_index": augmentation_index,
                        "rotation_index": view.rotation_index,
                        "features": features.detach().cpu(),
                    }
                )
                progress.advance()
        return views

    @torch.no_grad()
    def score_batch(self, images: torch.Tensor) -> np.ndarray:
        if self.velocity_model is None or self.operating_time is None:
            raise RuntimeError("a fitted checkpoint must be loaded before inference")
        features = self._normalize_and_average(self._extract_raw(images))
        query = features.reshape(-1, features.shape[-1]).to(self.device)
        noise = torch.randn_like(query)
        time_value = float(self.operating_time)
        state = time_value * query + (1.0 - time_value) * noise
        times = torch.full(
            (len(query), 1), time_value, device=self.device, dtype=torch.float32
        )
        with torch.amp.autocast(device_type="cuda", enabled=True):
            velocity = self.velocity_model(state, times)
        recovery = state + (1.0 - time_value) * velocity.float()
        scores = (query - recovery).square().sum(dim=-1)
        return (
            scores.reshape(len(images), PROTOCOL.grid_size, PROTOCOL.grid_size)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def save(self, path: str | Path) -> None:
        if self.velocity_model is None or self.operating_time is None:
            raise RuntimeError("cannot save an unfitted model")
        statistics = {
            layer: {"mean": mean.cpu(), "std": standard_deviation.cpu()}
            for layer, (mean, standard_deviation) in self.layer_statistics.items()
        }
        torch.save(
            {
                "flow": self.velocity_model.state_dict(),
                "statistics": statistics,
                "t": float(self.operating_time),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu")
        if set(state) != {"flow", "statistics", "t"}:
            raise RuntimeError(
                "checkpoint does not match the minimal Flownomaly format"
            )
        self.layer_statistics = {
            int(layer): (values["mean"].float(), values["std"].float())
            for layer, values in state["statistics"].items()
        }
        self.velocity_model = VelocityNetwork().to(self.device)
        self.velocity_model.load_state_dict(state["flow"])
        self.velocity_model.eval()
        for parameter in self.velocity_model.parameters():
            parameter.requires_grad_(False)
        self.operating_time = float(state["t"])
