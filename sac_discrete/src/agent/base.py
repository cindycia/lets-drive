import Pyro4
import torch
import time
from decimal import Decimal
from pathlib import Path
import os
import sys

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from utils import print_flush, error_handler
from agent.utils import to_batch


class BaseAgent:

    def __init__(self):
        self.env = None
        self.device = None
        self.shared_memory = None
        self.shared_weights = dict()
        self.memory = None
        self.virtual_memory = None
        self.replay_service = None
        self.virtual_replay_service = None
        self.critic = None
        self.value = None
        self.per = False

    def run(self):
        raise Exception('You need to implement run method.')

    def act(self, state):
        raise Exception('You need to implement act method.')

    def explore(self, state):
        raise Exception('You need to implement explore method.')

    def exploit(self, state):
        raise Exception('You need to implement explore method.')

    def interval(self):
        raise Exception('You need to implement interval method.')

    def calc_current_q(self, states, semantic_states, actions, rewards,
                       collisions, next_states, next_semantic_states, ncol_values, col_values, dones):
        raise Exception('You need to implement calc_current_q method.')

    def calc_target_q(self, states, semantic_states, actions, rewards,
                      collisions, next_states, next_semantic_states, ncol_values, col_values, dones):
        raise Exception('You need to implement calc_current_q method.')

    def calc_current_v(self, states, semantic_states, actions, rewards,
                       collisions, next_states, next_semantic_states, ncol_values, col_values, dones):
        raise Exception('You need to implement calc_current_v method.')

    def calc_target_v(self, states, semantic_states, actions, rewards,
                      collisions, next_states, next_semantic_states, ncol_values, col_values, dones):
        raise Exception('You need to implement calc_target_v method.')

    def calc_v_error(self, curr_v, curr_col, target_v, target_col):
        raise Exception('You need to implement cal_v_error method.')

    def calc_mask_error(self, cur_v_mask, cur_col_mask, target_v_mask, target_col_mask):
        raise Exception('You need to implement calc_mask_error method.')

    def load_weights(self):
        raise Exception('You need to implement load_weights method.')

    def save_weights(self):
        raise Exception('You need to implement save_weights method.')

    def update_params(self, optim, network, loss, grad_clip=None):
        mean_grads = None
        optim.zero_grad()
        loss.backward(retain_graph=False)
        if grad_clip is not None:
            for p in network.modules():
                torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
        if network is not None:
            mean_grads = self.calc_mean_grads(network)
        optim.step()
        return mean_grads

    def calc_mean_grads(self, network):
        total_grads = 0
        for m in network.modules():
            for p in m.parameters():
                total_grads += p.grad.clone().detach().sum()
        return total_grads / network.num_params

    def calc_grad_mag(self, network):
        n = 0
        grad_norm = 0
        for m in network.modules():
            for p in m.parameters():
                if p.grad is not None:
                    grad_norm = (n * grad_norm + p.grad.clone().detach().norm()) / (n + 1)
                    n = n + 1
        return grad_norm

    def total_data_count(self):
        try:
            count = self.replay_service.total_data_count()
            if self.virtual_replay_service:
                count += self.virtual_replay_service.total_data_count()
        except Exception as e:
            error_handler(e)
        return count

    def load_memory(self):
        try:
            self.last_replay_block, new_memory = self.replay_service.fetch_memory(self.last_replay_block)
            while new_memory is not None:
                # print_flush('[leaner_base] new_memory.keys()={}'.format(new_memory.keys()))
                self.memory.load(new_memory)
                self.last_replay_block, new_memory = self.replay_service.fetch_memory(self.last_replay_block)
        except Exception as e:
            error_handler(e)

        try:
            if self.virtual_memory is not None:
                self.last_virtual_replay_block, new_memory = \
                    self.virtual_replay_service.fetch_memory(self.last_virtual_replay_block)
                while new_memory is not None:
                    # print_flush('[leaner_base] virtual new_memory.keys()={}'.format(new_memory.keys()))
                    self.virtual_memory.load(new_memory)
                    self.last_virtual_replay_block, new_memory = \
                        self.virtual_replay_service.fetch_memory(self.last_virtual_replay_block)
        except Exception as e:
            error_handler(e)

    def save_memory(self, actor_id):
        print_flush("[actor_base] save_memory")

        if len(self.memory) > 1:
            key = '{:.9f}'.format(Decimal(time.time()))
            self.replay_service.add_memory(key, self.memory.get(), actor_id)
        self.memory.reset()

    def update_memory(self, state, next_state, offpolicy_action, semantic_state, next_semantic_state, factored_value,
                      terminal, reward, collision, episode_done, true_actor):
        try:
            if episode_done:
                self.memory.clear_multi_step_buff()
                if next_state is None:
                    return

            if reward is None:
                raise Exception('[agent.based] Warning: reward is {}} at non-terminal state in update_memory: '
                            'next_state {}, action {}, terminal {}, episode_done {}'.format(
                    reward, next_state, offpolicy_action, terminal, episode_done))

            clipped_reward = max(min(reward, 1.0), -1.0)
            if self.per:
                batch = to_batch(
                    state, semantic_state, offpolicy_action, clipped_reward, collision,
                    next_state, next_semantic_state, factored_value, terminal, self.device)

                if not true_actor:  # lets-drive-labeller
                    assert self.value is not None
                    with torch.no_grad():
                        curr_v, curr_col, cur_v_mask, cur_col_mask, _, _ = self.calc_current_v(*batch)
                    target_v, target_col, target_v_mask, target_col_mask = self.calc_target_v(*batch)
                    # mask_accuracy = min(((cur_col_mask >= 0.5).float() == target_col_mask).float().mean(),
                    #                     ((cur_v_mask >= 0.5).float() == target_v_mask).float().mean())
                    # error = self.calc_v_error(curr_v, curr_col, target_v, target_col).item()
                    error = self.calc_mask_error(cur_v_mask, cur_col_mask, target_v_mask, target_col_mask).item()
                else:  # lets-drive, imitation
                    if self.critic is not None:
                        with torch.no_grad():
                            curr_q1, curr_q2 = self.calc_current_q(*batch)
                        target_q = self.calc_target_q(*batch)
                        error = torch.abs(curr_q1 - target_q).item()
                    else:
                        assert self.value is not None
                        with torch.no_grad():
                            curr_v, curr_col, cur_v_mask, cur_col_mask, _, _ = self.calc_current_v(*batch)
                        target_v, target_col, target_v_mask, target_col_mask = self.calc_target_v(*batch)
                        error = self.calc_mask_error(cur_v_mask, cur_col_mask, target_v_mask, target_col_mask).item()

                self.memory.append(
                    state, semantic_state, offpolicy_action, clipped_reward, collision, next_state, next_semantic_state,
                    factored_value, terminal, error, episode_done=episode_done)
            else:
                self.memory.append(
                    state, semantic_state, offpolicy_action, clipped_reward, collision, next_state, next_semantic_state,
                    factored_value, terminal, episode_done=episode_done)
        except Exception as e:
            error_handler(e)

