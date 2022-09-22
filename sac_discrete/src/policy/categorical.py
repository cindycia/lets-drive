import torch
from torch.distributions import Categorical

import sys, os
from pathlib import Path
ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
sys.path.append(str(ws_root/'sac_discrete'/'src'))

from network import BaseNetwork, create_linear_network, create_dqn_base, create_resnet_base


class ConvCategoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_semantic_input, output_dim, initializer='kaiming'):
        super(ConvCategoricalPolicy, self).__init__()
        # self.base = create_dqn_base(num_channels)
        # num_features = 1024
        self.base = create_resnet_base(num_channels, num_blocks=2, num_hidden_planes=32)
        num_features = 4096
        self.fc = create_linear_network(
                num_features + num_semantic_input, output_dim, hidden_units=[512],
                output_activation='softmax',
                initializer=initializer)

    def forward(self, states, semantic_states):
        states = states / 255.0
        features = self.base(states)
        features = torch.cat((features, semantic_states), 1)
        action_probs = self.fc(features)
        return action_probs

    def sample(self, state, semantic_states):
        action_probs = self.forward(state, semantic_states)
        greedy_actions = torch.argmax(action_probs, dim=1, keepdim=True)

        categorical = Categorical(action_probs)
        actions = categorical.sample().view(-1, 1)

        log_action_probs = torch.log(
            action_probs + (action_probs == 0.0).float() * 1e-8)

        return actions, action_probs, log_action_probs, greedy_actions
