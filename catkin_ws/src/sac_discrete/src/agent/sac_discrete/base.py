from copy import deepcopy
import torch
from torch import nn

from ..base import BaseAgent
from utils import error_handler

class SacDiscreteAgent(BaseAgent):

    def __init__(self):
        super(SacDiscreteAgent, self).__init__()
        self.writer = None
        self.gamma_n = None
        self.alpha = None
        self.tau = None
        self.start_steps = None
        self.steps = None
        self.policy = nn.Sequential()
        self.critic = nn.Sequential()
        self.critic_target = nn.Sequential()

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

    def calc_current_q(self, states, semantic_states, actions, rewards, collisions, next_states, next_semantic_states,
                       ncol_values, col_values, dones):
        try:
            curr_q1, curr_q2 = self.critic(states, semantic_states)
            curr_q1 = curr_q1.gather(1, actions.long())
            curr_q2 = curr_q2.gather(1, actions.long())
            return curr_q1, curr_q2
        except Exception as e:
            error_handler(e)

    def calc_target_q(self, states, semantic_states, actions, rewards, collisions, next_states, next_semantic_states,
                      ncol_values, col_values, dones):
        try:
            with torch.no_grad():
                next_actions, next_action_probs, log_next_action_probs, _ =\
                    self.policy.sample(next_states, next_semantic_states)
                next_q1, next_q2 = self.critic_target(next_states, next_semantic_states)
                next_q = torch.min(next_q1, next_q2)
                next_q = next_action_probs * (
                    next_q - self.alpha * log_next_action_probs)
                next_q = next_q.mean(dim=1).unsqueeze(-1)

                target_q = rewards + (1.0 - dones) * self.gamma_n * next_q
            return target_q
        except Exception as e:
            error_handler(e)

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
                target_ncol = ncol_values
                target_col = col_values
                target_ncol_mask = (ncol_values < 0.0).float()
                target_col_mask = (col_values > 0.0).float()
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
