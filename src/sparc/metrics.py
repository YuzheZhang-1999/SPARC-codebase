"""Dataset-level confusion counts, Dice and intersection over union."""

import torch


class SegmentationMetrics:
    """Accumulate exact counts across batches; exclude background from macro scores."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = prediction.detach().cpu().long().reshape(-1)
        target = target.detach().cpu().long().reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        if ((target < 0) | (target >= self.num_classes)).any():
            raise ValueError("Target contains invalid class IDs")
        if ((prediction < 0) | (prediction >= self.num_classes)).any():
            raise ValueError("Prediction contains invalid class IDs")
        counts = torch.bincount(
            target * self.num_classes + prediction, minlength=self.num_classes**2
        )
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        matrix = self.confusion.double()
        true_positive = matrix.diag()
        false_positive = matrix.sum(0) - true_positive
        false_negative = matrix.sum(1) - true_positive
        dice_denominator = 2 * true_positive + false_positive + false_negative
        iou_denominator = true_positive + false_positive + false_negative
        # An absent class with no false positives receives 1, documented in README.
        dice = torch.where(dice_denominator > 0, 2 * true_positive / dice_denominator, 1.0)
        iou = torch.where(iou_denominator > 0, true_positive / iou_denominator, 1.0)
        return {
            "dice": dice[1:].mean().item(),
            "iou": iou[1:].mean().item(),
            "dice_per_class": dice.tolist(),
            "iou_per_class": iou.tolist(),
            "pixels": int(matrix.sum().item()),
            "confusion_matrix": self.confusion.tolist(),
        }
