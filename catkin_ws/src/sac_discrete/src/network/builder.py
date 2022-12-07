import torch
import torch.nn as nn
from .resnet import BasicBlock, normalization

str_to_initializer = {
    'uniform': nn.init.uniform_,
    'normal': nn.init.normal_,
    'eye': nn.init.eye_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'he': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_
    }


str_to_activation = {
    'elu': nn.ELU(),
    'hardshrink': nn.Hardshrink(),
    'hardtanh': nn.Hardtanh(),
    'leakyrelu': nn.LeakyReLU(),
    'logsigmoid': nn.LogSigmoid(),
    'prelu': nn.PReLU(),
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'rrelu': nn.RReLU(),
    'selu': nn.SELU(),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'logsoftmax': nn.LogSoftmax(),
    'softshrink': nn.Softshrink(),
    'softsign': nn.Softsign(),
    'tanh': nn.Tanh(),
    'tanhshrink': nn.Tanhshrink(),
    'softmin': nn.Softmin(),
    'softmax': nn.Softmax(dim=1),
    'none': None
    }


def initialize_weights(initializer):
    def initialize(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            initializer(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    return initialize


def create_linear_network(input_dim, output_dim, hidden_units=[],
                          hidden_activation='relu', output_activation=None,
                          initializer='xavier_uniform'):
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        model.append(str_to_activation[hidden_activation])
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(str_to_activation[output_activation])

    return nn.Sequential(*model).apply(
        initialize_weights(str_to_initializer[initializer]))


def create_dqn_base(num_channels, initializer='xavier_uniform'):
    return nn.Sequential(
        nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        Flatten(),
    ).apply(initialize_weights(str_to_initializer[initializer]))


def create_resnet_base(num_channels, num_blocks, num_hidden_planes, hidden_activation='leakyrelu',
                       initializer='xavier_uniform'):
    model = []
    # down-sampling kernel
    model.append(nn.Conv2d(num_channels, num_hidden_planes,
                           kernel_size=7, stride=2, padding=3, bias=False))
    model.append(normalization(code='no_norm', planes=num_hidden_planes))
    model.append(str_to_activation[hidden_activation])
    # residual tower
    for block in range(num_blocks):
        model.append(
            BasicBlock(num_hidden_planes, num_hidden_planes, stride=1,
                       initializer=str_to_initializer[initializer],
                       activation=str_to_activation[hidden_activation],
                       norm_code='no'))
    # 1*1 kernel
    model.append(nn.Conv2d(num_hidden_planes, 4, kernel_size=1, stride=1, bias=False))
    model.append(str_to_activation[hidden_activation])
    # flatten to 1D tensor
    model.append(Flatten())

    return nn.Sequential(*model).apply(
        initialize_weights(str_to_initializer[initializer]))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

