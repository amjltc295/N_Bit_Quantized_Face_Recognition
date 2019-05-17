from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss_fn(output, target)
