# Based on https://github.com/eladhoffer/convNet.pytorch
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction

QParams = namedtuple('QParams', ['s', 'z', 'num_bits', 'a', 'b'])


def calculate_qparams(x, num_bits):
    with torch.no_grad():
        a = x.min().item()
        b = x.max().item()

        level = 2 ** num_bits - 1
        s = (b - a) / level
        z = round((0.0 - a) / s)
        return QParams(a=a, b=b, s=s, z=z, num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, qparams=None, num_bits=None, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            qparams = calculate_qparams(input, num_bits=num_bits)

        a = qparams.a
        b = qparams.b
        s = qparams.s
        with torch.no_grad():
            # quantize
            output.clamp_(a, b).add_(-a).div_(s).round_()
            # dequantize
            output.mul_(s).add_(a)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


def quantize(
    x, qparams=None, num_bits=None, inplace=False
):
    return UniformQuantize().apply(x, qparams, num_bits, inplace)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, shape_measure=(1,), inplace=False, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_a', torch.zeros(*shape_measure))
        self.register_buffer('running_b', torch.zeros(*shape_measure))
        self.momentum = momentum
        self.inplace = inplace
        self.num_bits = num_bits

    def forward(self, input, qparams=None):

        if self.training:
            if qparams is None:
                qparams = calculate_qparams(input, num_bits=self.num_bits)
            with torch.no_grad():
                momentum = self.momentum
                self.running_a.mul_(momentum).add_(qparams.a * (1 - momentum))
                self.running_b.mul_(momentum).add_(qparams.b * (1 - momentum))
        else:
            qparams = QParams(a=self.running_a, b=self.running_b, num_bits=self.num_bits)
        q_input = quantize(input, qparams=qparams, inplace=self.inplace)
        return q_input


class QConv2d(nn.Conv2d):
    """ Quantized 2D Convolution """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            num_bits=8,
            num_bits_bias=32):

        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_bias = num_bits_bias
        self.quantize_input = QuantMeasure(self.num_bits, shape_measure=(1, 1, 1, 1))

    def forward(self, input):
        qinput = self.quantize_input(input)
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits)
        qweight = quantize(self.weight, qparams=weight_qparams)

        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_bias)
        else:
            qbias = None
        output = F.conv2d(qinput, qweight, qbias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output
