import torch.nn as nn


class CrossEntropyLossModule(nn.Module):
    def __init__(self, sum_label_dim=True):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.sum_label_dim = sum_label_dim

    def forward(self, pred, label, average=True):
        loss = self.loss(pred, label)
        if self.sum_label_dim:
            loss = loss.sum(dim=1)
        if average:
            loss = loss.mean()
        output = {"loss": loss}
        return output
