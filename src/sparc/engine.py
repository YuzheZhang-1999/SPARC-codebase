"""Teacher pre-training and teacher-guided joint decoder optimization."""

import logging
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from sparc.checkpoint import load_teacher, read_checkpoint, save_checkpoint
from sparc.config import resolve_device, seed_everything, write_json
from sparc.data.dataset import BSEADataset
from sparc.losses import SPARCLoss, SpectralDistillationLoss
from sparc.metrics import SegmentationMetrics
from sparc.models import SPARC, SpectralTeacher

LOGGER = logging.getLogger(__name__)


def make_loader(dataset, config: dict, training: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(config.get("seed", 0))
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=training,
        num_workers=config.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    *,
    stage: str,
    teacher=None,
    optimizer=None,
    accumulation_steps: int = 1,
) -> dict:
    training = optimizer is not None
    model.train(training)
    if teacher is not None:
        teacher.eval()
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    if training:
        optimizer.zero_grad(set_to_none=True)
    totals, count = {}, 0
    metrics = SegmentationMetrics(model.model_config["num_classes"]) if stage == "joint" else None
    with torch.set_grad_enabled(training):
        for index, sample in enumerate(loader):
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in sample.items()
            }
            raw = batch["raw"]
            if stage == "teacher":
                prediction = model(raw, batch["response"])
                losses = criterion(prediction, batch["spectral_target"])
                losses["total"] = losses["distillation"]
            else:
                with torch.no_grad():
                    target = (
                        teacher(raw, batch["response"])
                        if teacher is not None
                        else batch["spectral_target"]
                    )
                output = model(raw)
                losses = criterion(output, batch["label"], target)
                metrics.update(output["sense"].argmax(dim=1), batch["label"])
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {stage} loss in batch {index}")
            batch_size = raw.shape[0]
            if training:
                # Weight by samples, including a short final accumulation group.
                group_start = (index // accumulation_steps) * accumulation_steps
                group_end = min(group_start + accumulation_steps, len(loader))
                group_samples = min(group_end * loader.batch_size, len(loader.dataset)) - (
                    group_start * loader.batch_size
                )
                (loss * batch_size / group_samples).backward()
                if index + 1 == group_end:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            count += batch_size
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * batch_size
    if not count:
        raise ValueError("DataLoader produced no samples")
    result = {name: value / count for name, value in totals.items()}
    if metrics is not None:
        result.update(metrics.compute())
    return result


def train(config: dict, *, stage: str = "joint", device_name: str = "auto"):
    seed_everything(config["training"].get("seed", 0))
    device = resolve_device(device_name)
    training = config["training"]
    if stage not in {"teacher", "joint"}:
        raise ValueError("stage must be teacher or joint")
    if training["epochs"] < 1 or training["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    data = config["data"]
    teacher = None
    if stage == "teacher":
        model = SpectralTeacher(spectral_bands=data["spectral_bands"], dim=config["teacher"]["dim"])
        criterion = SpectralDistillationLoss(**config["loss"])
        target, response, labels = "hsi", True, False
        output_dir = Path(config["teacher"]["output_dir"])
        kind = "teacher"
    else:
        model = SPARC(**config["model"])
        criterion = SPARCLoss(**config["loss"])
        online = config["teacher"]["source"] == "online"
        if online:
            teacher_path = config["teacher"].get("checkpoint")
            if not teacher_path or not Path(teacher_path).is_file():
                raise FileNotFoundError(
                    "Run sparc pretrain-teacher first or configure a teacher checkpoint"
                )
            teacher = load_teacher(teacher_path, device)
            teacher_data = read_checkpoint(teacher_path)["config"]["data"]
            for key in ("raw_normalization", "spectral_scale"):
                if teacher_data.get(key) != data.get(key):
                    raise ValueError(f"Teacher and decoder data disagree on {key}")
            if teacher.model_config["spectral_bands"] != data["spectral_bands"]:
                raise ValueError("Teacher spectral-band count does not match the decoder")
        target, response, labels = (None if online else "cached"), online, True
        output_dir = Path(training["output_dir"])
        kind = "sparc"
    model = model.to(device)
    datasets = [
        BSEADataset(
            data,
            split,
            labels=labels,
            target=target,
            response=response,
            augment=(split == "train" and data.get("augment", True)),
        )
        for split in ("train", "val")
    ]
    loaders = [
        make_loader(dataset, training, training=(index == 0))
        for index, dataset in enumerate(datasets)
    ]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training["learning_rate"], betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training["scheduler_factor"],
        patience=training["scheduler_patience"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("best.pt", "last.pt", "history.json")):
        raise FileExistsError(f"{output_dir} already contains a run; choose a new output_dir")
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    history, best = [], math.inf
    for epoch in range(1, training["epochs"] + 1):
        train_result = run_epoch(
            model,
            loaders[0],
            criterion,
            device,
            stage=stage,
            teacher=teacher,
            optimizer=optimizer,
            accumulation_steps=training.get("accumulation_steps", 1),
        )
        val_result = run_epoch(model, loaders[1], criterion, device, stage=stage, teacher=teacher)
        scheduler.step(val_result["total"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_result,
            "val": val_result,
        }
        history.append(record)
        metadata = dict(
            config=config,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            validation=val_result,
        )
        if val_result["total"] < best:
            best = val_result["total"]
            save_checkpoint(output_dir / "best.pt", model, kind, **metadata)
        save_checkpoint(output_dir / "last.pt", model, kind, **metadata)
        write_json(output_dir / "history.json", history)
        LOGGER.info(
            "epoch %d/%d train=%.6f val=%.6f lr=%.3g",
            epoch,
            training["epochs"],
            train_result["total"],
            val_result["total"],
            optimizer.param_groups[0]["lr"],
        )
    return output_dir / "best.pt"
