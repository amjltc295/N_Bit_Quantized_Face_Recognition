# Based on https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
import torch.nn as nn
import math

from .components.quantization import QConv2d, QBatchNorm2d, QReLU6, QLinear


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        QConv2d(inp, oup, 3, stride, 1, bias=False),
        QBatchNorm2d(oup),
        QReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        QConv2d(inp, oup, 1, 1, 0, bias=False),
        QBatchNorm2d(oup),
        QReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                QConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                QBatchNorm2d(hidden_dim),
                QReLU6(inplace=True),
                # pw-linear
                QConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                QBatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                QConv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                QBatchNorm2d(hidden_dim),
                QReLU6(inplace=True),
                # dw
                QConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                QBatchNorm2d(hidden_dim),
                QReLU6(inplace=True),
                # pw-linear
                QConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                QBatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class QuantizedMobileNetV2(nn.Module):
    def __init__(self, n_class=512, input_size=224, width_mult=1., input_channel=32, last_channel=1024):
        super(QuantizedMobileNetV2, self).__init__()
        block = InvertedResidual
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            QLinear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        # No classifier as a backbone
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, QConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, QBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, QLinear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
