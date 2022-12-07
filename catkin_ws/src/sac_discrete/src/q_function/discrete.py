import os
import sys
from pathlib import Path

import torch

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from network import BaseNetwork, create_dqn_base, \
    create_linear_network


class DiscreteConvQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_semantic_input, output_dim, initializer='xavier'):
        super(DiscreteConvQNetwork, self).__init__()

        self.base = create_dqn_base(num_channels, initializer=initializer)
        self.V_stream = create_linear_network(
            1024 + num_semantic_input, 1, hidden_units=[512], initializer=initializer)
        self.A_stream = create_linear_network(
            1024 + num_semantic_input, output_dim, hidden_units=[512], initializer=initializer)

    def forward(self, states, semantic_states):
        h = self.base(states)
        h = torch.cat((h, semantic_states), 1)
        V = self.V_stream(h)
        A = self.A_stream(h)
        Q = V + A - A.mean(1, keepdim=True)
        return Q


class TwinedDiscreteConvQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_semantic_input, output_dim, initializer='xavier'):
        super(TwinedDiscreteConvQNetwork, self).__init__()

        self.Q1 = DiscreteConvQNetwork(
            num_channels, num_semantic_input, output_dim, initializer)
        self.Q2 = DiscreteConvQNetwork(
            num_channels, num_semantic_input, output_dim, initializer)

    def forward(self, states, semantic_states):
        states = states / 255.0
        Q1 = self.Q1(states, semantic_states)
        Q2 = self.Q2(states, semantic_states)
        return Q1, Q2
