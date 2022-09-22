import os
import sys
from pathlib import Path

import torch

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from network import BaseNetwork, create_dqn_base, create_resnet_base, \
    create_linear_network
from utils import print_flush, error_handler


class ConvVNetwork(BaseNetwork):
    def __init__(self, num_channels, num_semantic_input, initializer='xavier'):
        super(ConvVNetwork, self).__init__()
        try:
            self.base = create_dqn_base(num_channels)
            num_features = 1024
            # self.base = create_resnet_base(num_channels, num_blocks=1, num_hidden_planes=32, initializer='he')
            # num_features = 4096
            # self.V_stream = create_linear_network(
            #     num_features + num_semantic_input, 1, hidden_units=[], initializer='xavier')
            # self.C_stream = create_linear_network(
            #     num_features + num_semantic_input, 1, hidden_units=[], initializer='xavier')

            self.values = create_linear_network(
                num_features + num_semantic_input, 2, hidden_units=[], initializer='xavier')
            self.masks = create_linear_network(
                num_features + num_semantic_input, 2, output_activation='sigmoid', hidden_units=[],
                initializer='xavier')
        except Exception as e:
            error_handler(e)

    def forward(self, states, semantic_states):
        h = self.base(states)
        h = torch.cat((h, semantic_states), 1)
        # V = self.V_stream(h)
        # C = self.C_stream(h)
        # return V, C
        raw = self.values(h)
        mask = self.masks(h)
        masked = raw * (mask >= 0.5)
        return masked[:, 0].unsqueeze(1), masked[:, 1].unsqueeze(1), \
               mask[:, 0].unsqueeze(1), mask[:, 1].unsqueeze(1), \
               raw[:, 0].unsqueeze(1), raw[:, 1].unsqueeze(1)


if __name__ == '__main__':
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    example_forward_input = (torch.rand(1, 5, 64, 64).to(device),
                             torch.rand(1, 4).to(device))
    value = ConvVNetwork(
        5, 4).to(device)
    filename = str(ws_root / 'crowd_pomdp_planner' / 'test_value_net.pt')
    ts_value = torch.jit.trace(value, example_forward_input)
    torch.jit.save(ts_value, filename)



