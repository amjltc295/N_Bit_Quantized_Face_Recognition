# Based on https://github.com/eladhoffer/convNet.pytorch
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
from torch._jit_internal import weak_script_method

QParams = namedtuple('QParams', ['s', 'z', 'num_bits', 'a', 'b'])


def calculate_s_and_z(a, b, num_bits):
    level = 2 ** num_bits - 1
    s = max((b - a) / level, 1e-8)
    z = round((0.0 - a) / s)
    return s, z


def calculate_qparams(x, num_bits):
    with torch.no_grad():
        a = x.min().item()
        b = x.max().item()
        s, z = calculate_s_and_z(a, b, num_bits)

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
            s, z = calculate_s_and_z(self.running_a.item(), self.running_b.item(), self.num_bits)
            qparams = QParams(a=self.running_a.item(), b=self.running_b.item(), num_bits=self.num_bits, s=s, z=z)
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
        self.quantize_and_measure = QuantMeasure(self.num_bits)

    def forward(self, input):
        # qinput = self.quantize_and_measure(input)
        qinput = input
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits)
        qweight = quantize(self.weight, qparams=weight_qparams)

        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_bias)
        else:
            qbias = None
        output = F.conv2d(qinput, qweight, qbias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output


class QLinear(nn.Linear):
    """ Quantized fully connected layer """

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_bias=32):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_bias = num_bits_bias
        self.quantize_and_measure = QuantMeasure(self.num_bits)

    def forward(self, input):
        # qinput = self.quantize_and_measure(input)
        qinput = input
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits)
        qweight = quantize(self.weight, qparams=weight_qparams)

        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_bias)
        else:
            qbias = None
        output = F.linear(qinput, qweight, qbias)
        return output


class QBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_bits=8, num_bits_bias=32):
        super(QBatchNorm2d, self).__init__(
            num_features, eps=1e-5, momentum=0.1, affine=True,
            track_running_stats=True)
        self.num_bits = num_bits
        self.num_bits_bias = num_bits_bias
        self.quantize_and_measure = QuantMeasure(self.num_bits)

    @weak_script_method
    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # qinput = self.quantize_and_measure(input)
        qinput = input
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits)
        qweight = quantize(self.weight, qparams=weight_qparams)

        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_bias)
        else:
            qbias = None
        return F.batch_norm(
            qinput, self.running_mean, self.running_var, qweight, qbias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class QReLU6(nn.Module):
    def __init__(self, num_bits=8, inplace=True):
        super().__init__()
        self.num_bits = num_bits
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            out = x.clamp_(0, 2 ** self.num_bits)
        else:
            out = torch.clamps(x, 0, 2 ** self.num_bits)
        qout = quantize(out, num_bits=self.num_bits)
        return qout
