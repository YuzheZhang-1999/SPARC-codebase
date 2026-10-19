"""The SPARC command-line interface; all commands are safe to import."""

import argparse
import hashlib
import json
import logging
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparc", description="SPARC spectral perception framework"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pretrain-teacher", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--device", default="auto")
        command.add_argument("--backend", choices=("mamba", "reference"))
    command = commands.add_parser("evaluate", help="Evaluate the sensing path on labelled data")
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--data-root", type=Path)
    command.add_argument("--split", choices=("train", "val", "test"), default="test")
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--device", default="auto")
    command.add_argument("--backend", choices=("mamba", "reference"))
    command = commands.add_parser(
        "predict", help="Predict class masks from a folder of RAW MAT files"
    )
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--input", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--device", default="auto")
    command.add_argument("--backend", choices=("mamba", "reference"))
    command = commands.add_parser("export", help="Extract a sensing-only PyTorch checkpoint")
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--backend", choices=("mamba", "reference"))
    command = commands.add_parser(
        "encode", help="Synthesize BSEA-RAW using an aligned fixed response"
    )
    command.add_argument("--cube", required=True, type=Path)
    command.add_argument("--response", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("cache-teacher", help="Generate spectral teacher targets offline")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--split", required=True, choices=("train", "val", "test"))
    command.add_argument("--device", default="auto")
    command = commands.add_parser(
        "import-pretrained", help="Convert encoder/decoder weights to an inference checkpoint"
    )
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--backend", choices=("mamba", "reference"), default="reference")
    return parser


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader

    from sparc.checkpoint import load_inference
    from sparc.config import resolve_device, write_json
    from sparc.data.dataset import BSEADataset
    from sparc.metrics import SegmentationMetrics

    device = resolve_device(args.device)
    model, payload = load_inference(args.checkpoint, device, args.backend)
    data_config = dict(payload["config"]["data"])
    if args.data_root:
        data_config["root"] = str(args.data_root.resolve())
    dataset = BSEADataset(data_config, args.split)
    metrics = SegmentationMetrics(model.model_config["num_classes"])
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1):
            prediction = model(batch["raw"].to(device)).argmax(dim=1)
            metrics.update(prediction, batch["label"])
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "samples": len(dataset),
        **metrics.compute(),
    }
    write_json(args.output, result)
    return result


def predict(args):
    import numpy as np
    import torch
    from PIL import Image

    from sparc.checkpoint import load_inference
    from sparc.config import resolve_device
    from sparc.data.io import normalize_raw, read_array

    device = resolve_device(args.device)
    model, payload = load_inference(args.checkpoint, device, args.backend)
    if model.model_config["num_classes"] > 65536:
        raise ValueError("PNG prediction supports up to 65536 classes")
    files = sorted(args.input.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat RAW files in {args.input}")
    outputs = [args.output / f"{file.stem}.png" for file in files]
    if any(file.exists() for file in outputs):
        raise FileExistsError("Prediction output exists; choose an empty output directory")
    mode = payload["config"]["data"].get("raw_normalization", "max")
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for file, destination in zip(files, outputs):
            raw = normalize_raw(read_array(file, "raw", 2), mode)
            logits = model(torch.from_numpy(raw)[None, None].to(device))
            mask = logits.argmax(1)[0].cpu().numpy()
            # Binary outputs are display masks; multiclass outputs retain class IDs.
            if model.model_config["num_classes"] == 2:
                mask = mask.astype(np.uint8) * 255
            else:
                dtype = np.uint8 if model.model_config["num_classes"] <= 256 else np.uint16
                mask = mask.astype(dtype)
            Image.fromarray(mask).save(destination)
    return {"predictions": len(outputs), "output": str(args.output)}


def export(args):
    import torch

    from sparc.checkpoint import load_inference, save_checkpoint

    if args.output.exists():
        raise FileExistsError(f"Checkpoint already exists: {args.output}")
    model, payload = load_inference(args.checkpoint, torch.device("cpu"), args.backend)
    save_checkpoint(
        args.output,
        model,
        "inference",
        config=payload["config"],
        epoch=payload.get("epoch", 0),
        source_checkpoint=str(args.checkpoint),
    )
    return {"output": str(args.output), "parameters": sum(p.numel() for p in model.parameters())}


def encode(args):
    import torch
    from scipy.io import savemat

    from sparc.data.io import read_array
    from sparc.optics import BSEAEncoder

    if args.output.exists():
        raise FileExistsError(f"RAW output already exists: {args.output}")
    cube = torch.from_numpy(read_array(args.cube, "cube", 3).astype("float32"))
    response = torch.from_numpy(read_array(args.response, "mask", 3).astype("float32"))
    if (cube < 0).any() or (response < 0).any():
        raise ValueError("Cube and physical BSEA response must be nonnegative")
    raw = BSEAEncoder()(cube[None], response)[0, 0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output, {"raw": raw.numpy()}, do_compression=True)
    return {"output": str(args.output), "shape": list(raw.shape)}


def cache_teacher(args):
    import torch
    from scipy.io import savemat
    from torch.utils.data import DataLoader

    from sparc.checkpoint import load_teacher, read_checkpoint
    from sparc.config import load_config, resolve_device, write_json
    from sparc.data.dataset import BSEADataset

    config = load_config(args.config)
    device = resolve_device(args.device)
    teacher = load_teacher(args.checkpoint, device)
    teacher_config = read_checkpoint(args.checkpoint)["config"]["data"]
    data_config = config["data"]
    for key in ("spectral_bands", "raw_normalization", "spectral_scale"):
        if teacher_config.get(key) != data_config.get(key):
            raise ValueError(f"Teacher and target data configuration disagree on {key}")
    dataset = BSEADataset(data_config, args.split, labels=False, response=True)
    output = Path(data_config["root"]) / "teacher_targets"
    outputs = [output / f"{sample_id}.mat" for sample_id in dataset.ids]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Some teacher targets already exist; existing targets are preserved")
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    provenance = {
        "teacher_sha256": digest,
        "raw_normalization": data_config["raw_normalization"],
        "spectral_scale": data_config["spectral_scale"],
    }
    metadata = output / "metadata.json"
    if metadata.exists() and json.loads(metadata.read_text()) != provenance:
        raise ValueError("Existing cached targets have a different teacher or normalization")
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1):
            target = teacher(batch["raw"].to(device), batch["response"].to(device))[0].cpu().numpy()
            savemat(output / f"{batch['id'][0]}.mat", {"cube": target}, do_compression=True)
    write_json(metadata, provenance)
    return {"cached_samples": len(dataset), "output": str(output)}


def import_pretrained(args):
    import torch

    from sparc.checkpoint import save_checkpoint
    from sparc.models.pretrained import PretrainedSPARCInference, map_pretrained_state

    if args.output.exists():
        raise FileExistsError(f"Checkpoint already exists: {args.output}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    mapped, excluded = map_pretrained_state(state)
    dim = state["stem.weight"].shape[0]
    classes = state["head.weight"].shape[0]
    model = PretrainedSPARCInference(dim=dim, num_classes=classes, mamba_backend=args.backend)
    model.load_state_dict(mapped, strict=True)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    config = {"data": {"raw_normalization": "max", "num_classes": classes}}
    save_checkpoint(
        args.output,
        model.eval(),
        "inference",
        config=config,
        source_sha256=digest,
        source_checkpoint=args.checkpoint.name,
        excluded_auxiliary_parameters=len(excluded),
    )
    return {
        "output": str(args.output),
        "architecture": "pretrained_v1",
        "sensing_tensors": len(mapped),
        "excluded_auxiliary_tensors": len(excluded),
        "source_sha256": digest,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command in {"train", "pretrain-teacher"}:
        from sparc.config import load_config
        from sparc.engine import train

        config = load_config(args.config)
        if args.backend:
            config["model"]["mamba_backend"] = args.backend
        result = {
            "best_checkpoint": str(
                train(
                    config,
                    stage="teacher" if args.command == "pretrain-teacher" else "joint",
                    device_name=args.device,
                )
            )
        }
    else:
        handlers = {
            "evaluate": evaluate,
            "predict": predict,
            "export": export,
            "encode": encode,
            "cache-teacher": cache_teacher,
            "import-pretrained": import_pretrained,
        }
        result = handlers[args.command](args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
