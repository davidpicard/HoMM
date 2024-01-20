import torch.nn as nn

class MSELossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target, average=True):
        loss = self.loss(pred, target).mean()
        output = {"loss": loss}
        return output