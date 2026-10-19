"""Structural, gradient and deployment tests with the CPU selective scan."""

import importlib.util
import unittest

import torch

from sparc.models import SPARC, SpectralTeacher
from sparc.models.mamba import ReferenceMamba

torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def test_training_outputs_and_shared_gradients(self):
        model = SPARC(dim=2, spectral_bands=4, mamba_backend="reference")
        raw = torch.rand(1, 1, 17, 19)
        calls = []
        handle = model.ssls_modulator.register_forward_hook(lambda *args: calls.append(1))
        outputs = model(raw)
        handle.remove()
        self.assertEqual(outputs["sense"].shape, (1, 2, 17, 19))
        self.assertEqual(outputs["recon"].shape, (1, 4, 17, 19))
        self.assertEqual(len(calls), 2)
        shared = next(model.ssls_modulator.parameters())
        for branch in ("sense", "recon"):
            gradient = torch.autograd.grad(
                outputs[branch].square().mean(), shared, retain_graph=True
            )[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_export_preserves_predictions_and_removes_auxiliary(self):
        model = SPARC(dim=2, spectral_bands=4, mamba_backend="reference").eval()
        raw = torch.rand(1, 1, 17, 19)
        inference = model.to_inference()
        with torch.no_grad():
            torch.testing.assert_close(model(raw)["sense"], inference(raw), rtol=0, atol=0)
        self.assertFalse(any("auxiliary" in key for key in inference.state_dict()))
        self.assertLess(
            sum(p.numel() for p in inference.parameters()),
            sum(p.numel() for p in model.parameters()),
        )
        self.assertIsNot(model.ssls_modulator, inference.ssls_modulator)

    def test_frozen_teacher_and_shape(self):
        teacher = SpectralTeacher(dim=2, spectral_bands=4).freeze()
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        output = teacher(torch.rand(1, 1, 17, 19), torch.rand(1, 4, 17, 19))
        self.assertEqual(output.shape, (1, 4, 17, 19))
        self.assertFalse(output.requires_grad)

    def test_reference_mamba_is_causal_and_differentiable(self):
        mamba = ReferenceMamba(4)
        tokens = torch.randn(2, 9, 4, requires_grad=True)
        altered = tokens.detach().clone()
        altered[:, 5:] += 10
        output = mamba(tokens)
        torch.testing.assert_close(output[:, :5], mamba(altered)[:, :5])
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(tokens.grad.abs().sum().item(), 0)

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("mamba_ssm"),
        "CUDA and mamba-ssm required for fused-backend parity",
    )
    def test_cuda_mamba_matches_reference(self):
        from mamba_ssm import Mamba

        reference = ReferenceMamba(8).cuda()
        fused = Mamba(d_model=8, d_state=16, d_conv=4, expand=2).cuda()
        fused.load_state_dict(reference.state_dict(), strict=True)
        tokens = torch.randn(2, 19, 8, device="cuda")
        torch.testing.assert_close(reference(tokens), fused(tokens), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
