import torch
import torch.optim as optim
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.fc = nn.Linear(216, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        out = self.fc(c1.view(-1, self.num_flat_features(c1)))
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def weight_quantize_parameter(weight, bits=8):
    max_w = weight.max().item()
    min_w = weight.min().item()
    level = 2 ** bits - 1
    scale = (max_w - min_w) / level
    zero_point = round((0.0 - min_w) / scale)
    return scale, zero_point, min_w, max_w


def quantize(weight, S, Z, a, b, bits=8):
    return ((torch.clamp(weight, a, b) - a) / S).round() * S + a


def unquantize(weight, S, Z):
    return S * (weight - Z)


def quantize_net(net, bits=8):
    for n, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            s, z, a, b = weight_quantize_parameter(module.weight, bits)
            module.weight = torch.nn.Parameter(quantize(module.weight, s, z, a, b, bits))
            if module.bias is not None:
                s, z, a, b = weight_quantize_parameter(module.bias, bits)
                module.bias = torch.nn.Parameter(quantize(module.bias, s, z, a, b, bits))


net = Net()
print(net)

input = torch.randn(1, 1, 10, 10)
out = net(input)
print(out)

quantize_net(net)
qout = net(input)
print(qout)


criterion = nn.MSELoss()

qdiff = criterion(out, qout)
print(f"Qdif {qdiff}")

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()

target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output

loss = criterion(out, target)
print(loss)
loss.backward()

optimizer.step()
out = net(input)
print(out)
loss = criterion(out, target)
print(loss)
