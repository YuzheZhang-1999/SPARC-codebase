"""Two-stage optimization, checkpoint export, evaluation and prediction tests."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.io import loadmat, savemat

from sparc.checkpoint import load_inference, read_checkpoint
from sparc.cli import main

torch.set_num_threads(1)


class PipelineTests(unittest.TestCase):
    def test_two_stage_training_export_evaluation_prediction_and_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            for folder in ("raw", "hsi", "gt"):
                (data / folder).mkdir(parents=True)
            generator = np.random.default_rng(0)
            cube = generator.random((4, 16, 16), dtype=np.float32)
            response = generator.random((4, 16, 16), dtype=np.float32)
            raw = (cube * response).sum(axis=0)
            savemat(data / "mask.mat", {"mask": response})
            for sample_id in ("a", "b", "c"):
                savemat(data / f"raw/{sample_id}.mat", {"raw": raw})
                savemat(data / f"hsi/{sample_id}.mat", {"cube": cube})
                savemat(data / f"gt/{sample_id}.mat", {"gt": (raw > raw.mean()).astype(np.int64)})
            (data / "splits.json").write_text(
                json.dumps({"train": ["a"], "val": ["b"], "test": ["c"]})
            )
            teacher_checkpoint = root / "teacher/best.pt"
            config = {
                "model": dict(dim=2, num_classes=2, spectral_bands=4, mamba_backend="reference"),
                "data": dict(
                    root=str(data),
                    spectral_bands=4,
                    num_classes=2,
                    raw_normalization="max",
                    spectral_scale=1.0,
                    augment=False,
                ),
                "teacher": dict(
                    source="online",
                    checkpoint=str(teacher_checkpoint),
                    dim=2,
                    output_dir=str(root / "teacher"),
                ),
                "training": dict(
                    output_dir=str(root / "joint"),
                    epochs=1,
                    batch_size=1,
                    accumulation_steps=4,
                    num_workers=0,
                    seed=0,
                    learning_rate=1e-4,
                    scheduler_factor=0.5,
                    scheduler_patience=10,
                ),
                "loss": dict(alpha=1.0, beta=0.2, tv_reduction="sum", epsilon=1e-6),
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            main(["pretrain-teacher", "--config", str(config_path), "--device", "cpu"])
            main(["train", "--config", str(config_path), "--device", "cpu"])
            checkpoint = root / "joint/best.pt"
            exported = root / "inference.pt"
            main(["export", "--checkpoint", str(checkpoint), "--output", str(exported)])
            full, _ = load_inference(checkpoint, torch.device("cpu"))
            inference, _ = load_inference(exported, torch.device("cpu"))
            test_raw = torch.from_numpy(raw / raw.max())[None, None]
            with torch.no_grad():
                torch.testing.assert_close(full(test_raw), inference(test_raw), rtol=0, atol=0)
            payload = read_checkpoint(exported)
            self.assertNotIn("optimizer", payload)
            self.assertFalse(any("auxiliary" in key for key in payload["model"]))
            metrics_path = root / "metrics.json"
            main(
                [
                    "evaluate",
                    "--checkpoint",
                    str(exported),
                    "--output",
                    str(metrics_path),
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(json.loads(metrics_path.read_text())["pixels"], 256)
            main(
                [
                    "predict",
                    "--checkpoint",
                    str(exported),
                    "--input",
                    str(data / "raw"),
                    "--output",
                    str(root / "predictions"),
                    "--device",
                    "cpu",
                ]
            )
            predicted = np.asarray(Image.open(root / "predictions/a.png"))
            self.assertEqual(predicted.shape, (16, 16))
            self.assertTrue(set(np.unique(predicted)).issubset({0, 255}))
            main(
                [
                    "cache-teacher",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(teacher_checkpoint),
                    "--split",
                    "train",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(loadmat(data / "teacher_targets/a.mat")["cube"].shape, (4, 16, 16))
            config["teacher"]["source"] = "cached"
            # Cache validation targets as well, then exercise cached training.
            main(
                [
                    "cache-teacher",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(teacher_checkpoint),
                    "--split",
                    "val",
                    "--device",
                    "cpu",
                ]
            )
            config["training"]["output_dir"] = str(root / "cached")
            config_path.write_text(yaml.safe_dump(config))
            main(["train", "--config", str(config_path), "--device", "cpu"])
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from sparc.cli import main; raise SystemExit(main())",
                    "encode",
                    "--cube",
                    str(data / "hsi/a.mat"),
                    "--response",
                    str(data / "mask.mat"),
                    "--output",
                    str(root / "encoded.mat"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            np.testing.assert_allclose(loadmat(root / "encoded.mat")["raw"], raw, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
