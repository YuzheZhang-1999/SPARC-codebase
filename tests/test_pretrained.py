"""Pretrained checkpoint serialization and state-dictionary mapping tests."""

import tempfile
import unittest
from pathlib import Path

import torch

from sparc.checkpoint import load_inference, save_checkpoint
from sparc.models.pretrained import PretrainedSPARCInference, map_pretrained_state


class PretrainedTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_trained_architecture(self):
        torch.set_num_threads(1)
        model = PretrainedSPARCInference(dim=2, mamba_backend="reference").eval()
        raw = torch.rand(1, 1, 16, 16)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "pretrained.pt"
            save_checkpoint(
                checkpoint, model, "inference", config={"data": {"raw_normalization": "max"}}
            )
            restored, _ = load_inference(checkpoint, torch.device("cpu"))
            self.assertIsInstance(restored, PretrainedSPARCInference)
            with torch.inference_mode():
                torch.testing.assert_close(model(raw), restored(raw), rtol=0, atol=0)

    def test_state_mapping_records_auxiliary_parameters_and_rejects_unknown_keys(self):
        state = {
            "stem.weight": torch.randn(2, 1, 3, 3),
            "embedding1.weight": torch.randn(2, 1, 3, 3),
        }
        mapped, excluded = map_pretrained_state(state)
        self.assertEqual(list(mapped), ["optical_adapter.weight"])
        self.assertEqual(excluded, ["embedding1.weight"])
        with self.assertRaises(ValueError):
            map_pretrained_state({"unrecognized.weight": torch.ones(1)})


if __name__ == "__main__":
    unittest.main()
