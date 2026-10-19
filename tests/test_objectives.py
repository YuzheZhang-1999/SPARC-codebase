import unittest

import torch

from sparc.losses import SPARCLoss, SpectralDistillationLoss, spatial_total_variation
from sparc.metrics import SegmentationMetrics
from sparc.optics import BSEAEncoder


class ObjectiveTests(unittest.TestCase):
    def test_optical_projection_and_gradient(self):
        cube = torch.ones(2, 4, 3, 5, requires_grad=True)
        response = torch.full((4, 3, 5), 0.5)
        raw = BSEAEncoder()(cube, response)
        torch.testing.assert_close(raw, torch.full((2, 1, 3, 5), 2.0))
        raw.sum().backward()
        torch.testing.assert_close(cube.grad, torch.full_like(cube, 0.5))

    def test_tv_uses_prediction_neighbours(self):
        prediction = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
        self.assertAlmostEqual(spatial_total_variation(prediction).item(), 1.0)
        prediction = torch.ones(1, 1, 3, 4, requires_grad=True)
        result = spatial_total_variation(prediction)
        self.assertEqual(result.item(), 0)
        result.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_identical_constant_spectra_have_zero_distillation(self):
        spectra = torch.ones(1, 4, 3, 4, requires_grad=True)
        target = spectra.detach().clone().requires_grad_()
        result = SpectralDistillationLoss()(spectra, target)
        self.assertEqual(result["distillation"].item(), 0)
        result["distillation"].backward()
        self.assertTrue(torch.isfinite(spectra.grad).all())
        self.assertIsNone(target.grad)

    def test_zero_target_is_numerically_stable(self):
        output = {
            "sense": torch.randn(2, 2, 4, 5, requires_grad=True),
            "recon": torch.zeros(2, 4, 4, 5, requires_grad=True),
        }
        target = torch.zeros_like(output["recon"])
        result = SPARCLoss()(output, torch.zeros(2, 4, 5, dtype=torch.long), target)
        result["total"].backward()
        self.assertTrue(torch.isfinite(output["recon"].grad).all())

    def test_metrics_accumulate_counts_across_batches(self):
        metrics = SegmentationMetrics(2)
        metrics.update(torch.tensor([1, 1]), torch.tensor([1, 0]))
        metrics.update(torch.tensor([0, 0]), torch.tensor([1, 0]))
        result = metrics.compute()
        self.assertAlmostEqual(result["dice"], 0.5)
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertEqual(result["pixels"], 4)


if __name__ == "__main__":
    unittest.main()
