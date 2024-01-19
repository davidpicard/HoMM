from torchmetrics import Metric
import torch


class ClassificationMetrics(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = preds.argmax(dim=1)
        target = target.argmax(dim=1)
        for c in range(self.num_classes):
            self.tp[c] += ((preds == c) & (target == c)).sum()
            self.fp[c] += ((preds == c) & (target != c)).sum()
            self.fn[c] += ((preds != c) & (target == c)).sum()
            self.tn[c] += ((preds != c) & (target != c)).sum()

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-8)

        return {
            "acc": acc.mean(),
            "f1": f1.mean(),
            "precision": precision.mean(),
            "recall": recall.mean(),
        }
