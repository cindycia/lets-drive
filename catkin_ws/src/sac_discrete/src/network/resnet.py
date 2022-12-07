import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Identity(nn.Module):
    def forward(self, x):
        return x


def normalization(code, planes):
    if code == 'batch':
        return nn.BatchNorm2d(planes, track_running_stats=True)
    elif code == 'instance':
        return nn.InstanceNorm2d(planes)
    else:
        return Identity()


def initialize_weights(initializer):
    def initialize(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            initializer(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return initialize


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, initializer, activation, stride=1, norm_code='instance'):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.model = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            normalization(norm_code, planes),
            activation,
            conv3x3(planes, planes),
            normalization(norm_code, planes)
        )
        self.model.apply(initialize_weights(initializer))

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        out = self.activation(out)
        return out


