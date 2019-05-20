import torch
import torch.nn as nn

from src.model.backbone.components.quantization import quantize, calculate_qparams


def test_qweight():
    conv = nn.Conv2d(1, 1, 2, bias=False)
    w = torch.Tensor([
        [0, 1.001],
        [1.999, 2.55]
    ])
    conv.weight = torch.nn.Parameter(w.view(1, 1, 2, 2))

    weight_qparams = calculate_qparams(conv.weight, num_bits=8)
    qweight = quantize(conv.weight, qparams=weight_qparams)

    target_qweight = torch.Tensor([
        [0, 1],
        [2, 2.55]
    ]).view(1, 1, 2, 2)

    criterion = nn.MSELoss()
    loss = criterion(qweight, target_qweight)
    assert loss.item() < 1e-10
