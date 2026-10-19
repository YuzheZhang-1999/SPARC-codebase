import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from sparc.data.dataset import BSEADataset
from sparc.data.io import normalize_raw


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ("raw", "hsi", "gt", "teacher_targets"):
            (self.root / name).mkdir()
        (self.root / "splits.json").write_text(json.dumps({"train": ["a"], "val": ["b"]}))
        pattern = np.arange(16, dtype=np.float32).reshape(4, 4) / 15
        self.pattern = pattern
        savemat(self.root / "raw/a.mat", {"raw": pattern})
        savemat(self.root / "gt/a.mat", {"gt": (pattern > 0.5).astype(np.int64)})
        cube = np.repeat(pattern[None], 4, axis=0)
        savemat(self.root / "hsi/a.mat", {"cube": cube})
        savemat(self.root / "teacher_targets/a.mat", {"cube": cube})
        savemat(self.root / "mask.mat", {"mask": cube})
        self.config = dict(
            root=str(self.root),
            spectral_bands=4,
            num_classes=2,
            raw_normalization="max",
            spectral_scale=1.0,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_augmentation_preserves_alignment_including_calibration(self):
        torch.manual_seed(0)
        sample = BSEADataset(self.config, "train", target="hsi", response=True, augment=True)[0]
        torch.testing.assert_close(sample["raw"][0], sample["spectral_target"][0])
        torch.testing.assert_close(sample["raw"][0], sample["response"][0])
        torch.testing.assert_close(sample["label"], (sample["raw"][0] > 0.5).long())

    def test_split_overlap_is_rejected(self):
        (self.root / "splits.json").write_text(json.dumps({"train": ["a"], "val": ["a.mat"]}))
        with self.assertRaises(ValueError):
            BSEADataset(self.config, "train")

    def test_single_class_labels_do_not_change_shape(self):
        savemat(self.root / "gt/a.mat", {"gt": np.zeros((4, 4), dtype=np.int64)})
        sample = BSEADataset(self.config, "train")[0]
        self.assertEqual(sample["label"].shape, (4, 4))
        self.assertEqual(sample["label"].dtype, torch.int64)

    def test_empty_raw_is_finite(self):
        raw = normalize_raw(np.zeros((4, 5)))
        self.assertTrue(np.isfinite(raw).all())
        self.assertEqual(raw.max(), 0)

    def test_cached_teacher_scale_is_not_applied_twice(self):
        self.config["spectral_scale"] = 10.0
        sample = BSEADataset(self.config, "train", target="cached")[0]
        torch.testing.assert_close(sample["spectral_target"][0], torch.from_numpy(self.pattern))


if __name__ == "__main__":
    unittest.main()
