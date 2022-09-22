from copy import deepcopy
import torch
from torch import nn

from ..base import BaseAgent
from utils import print_flush, error_handler

class ImitationAgent(BaseAgent):

    def __init__(self):
        super(ImitationAgent, self).__init__()
        self.writer = None
        self.gamma_n = None
        self.alpha = None
        self.tau = None
        self.start_steps = None
        self.steps = None
        self.policy = nn.Sequential()
        self.value = nn.Sequential()
        self.value_target = nn.Sequential()

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        state = \
            torch.ByteTensor(state[None, ...]).to(self.device).float()
        with torch.no_grad():
            action, _, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        state = \
            torch.ByteTensor(state[None, ...]).to(self.device).float()
        with torch.no_grad():
            _, _, _, action = self.policy.sample(state)
        return action.item()

    def calc_current_v(self, states, semantic_states, actions, rewards, collisions, next_states, next_semantic_states,
                       ncol_values, col_values, dones):
        try:
            ncol_value, col_value, ncol_mask, col_mask, ncol_raw, col_raw = self.value(states, semantic_states)
            return ncol_value, col_value, ncol_mask.float(), col_mask.float(), ncol_raw, col_raw
        except Exception as e:
            error_handler(e)

    def calc_target_v(self, states, semantic_states, actions, rewards, collisions, next_states, next_semantic_states,
                      ncol_values, col_values, dones):
        try:
            with torch.no_grad():
                # next_v, next_col = self.value_target(next_states, next_semantic_states)
                # target_v = rewards + (1.0 - dones) * self.gamma_n * next_v
                # target_col = collisions + (1.0 - dones) * self.gamma_n * next_col
                target_ncol = ncol_values
                target_col = col_values
                target_ncol_mask = (ncol_values < 0.0).float()
                target_col_mask = (col_values > 0.0).float()

                # l=5
                # print_flush('target_col={}\ntarget_col_mask={}'.format(target_col[:l], target_col_mask[:l]))

            return target_ncol, target_col, target_ncol_mask, target_col_mask
        except Exception as e:
            error_handler(e)

    def calc_v_error(self, curr_v, curr_col, target_v, target_col):
        try:
            return torch.max(torch.abs(curr_v.detach() - target_v),
                               torch.abs(curr_col.detach() - target_col))
        except Exception as e:
            error_handler(e)

    def calc_mask_error(self, cur_v_mask, cur_col_mask, target_v_mask, target_col_mask):
        try:
            return torch.ones(cur_v_mask.shape)
            # return 0.2 + torch.max(((cur_col_mask >= 0.5).float() != target_col_mask).float(),
            #                        ((cur_v_mask >= 0.5).float() != target_v_mask).float())
        except Exception as e:
            error_handler(e)

    def load_weights(self):
        try:
            raise Exception('method not implemented')
        except KeyError:
            return False

    def save_weights(self):
        raise Exception('method not implemented')

    def __del__(self):
        # if self.writer:
        #     self.writer.close()
        self.env.close()
