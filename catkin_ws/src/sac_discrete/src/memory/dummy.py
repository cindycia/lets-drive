import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler

from .multi_step import MultiStepBuff
from utils import print_flush, error_handler


class DummyMemory(dict):
    state_keys = ['state', 'semantic_state', 'next_state', 'next_semantic_state']
    np_keys = ['action', 'reward', 'done', 'collision', 'factored_value']
    keys = state_keys + np_keys

    def __init__(self, capacity, state_shape, semantic_state_shape, action_shape, device):
        super(DummyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.semantic_state_shape = semantic_state_shape
        self.action_shape = action_shape
        self.device = device
        self.reset()

    def size(self):
        return len(self['state'])

    def reset(self):
        for key in self.state_keys:
            self[key] = []

        self['action'] = np.empty(
            (self.capacity, *self.action_shape), dtype=np.float32)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['collision'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['factored_value'] = np.empty((self.capacity, 2), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, semantic_state, action, reward, collision, next_state, next_semantic_state, factored_value,
               done, episode_done=None):
        self._append(
            state, semantic_state, action, reward, collision, next_state, next_semantic_state, factored_value, done)

    def _append(self, state, semantic_state, action, reward, collision, next_state, next_semantic_state, factored_value,
                done):
        try:
            self['state'].append(state)
            self['next_state'].append(next_state)
            self['semantic_state'].append(semantic_state)
            self['next_semantic_state'].append(next_semantic_state)
            self['action'][self._p] = action
            self['reward'][self._p] = reward
            self['done'][self._p] = done
            self['collision'][self._p] = float(collision)
            self['factored_value'][self._p][:] = factored_value

            self._n = min(self._n + 1, self.capacity)
            self._p = (self._p + 1) % self.capacity

            self.truncate()
        except Exception as e:
            error_handler(e)

    def revise_latest(self, flag, new_data):
        num = len(new_data)
        if num > self._p:
            tail_num = self._p
            self[flag][:tail_num] = new_data[-tail_num:]
            head_num = num - self._p
            self[flag][-head_num:] = new_data[:head_num]
        else:
            self[flag][self._p - num:self._p] = new_data

    def truncate(self):
        if len(self) > self.capacity:
            for key in self.state_keys:
                self[key] = self[key][-self.capacity:]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)

        semantic_states = np.empty(
            (batch_size, self.semantic_state_shape), dtype=np.float32)
        semantic_next_states = np.empty(
            (batch_size, self.semantic_state_shape), dtype=np.float32)

        for i, index in enumerate(indices):
            _index = np.mod(index+bias, self.capacity)
            states[i, ...] = \
                np.array(self['state'][_index], dtype=np.uint8)
            next_states[i, ...] = \
                np.array(self['next_state'][_index], dtype=np.uint8)
            semantic_states[i, ...] = \
                np.array(self['semantic_state'][_index], dtype=np.float32)
            semantic_next_states[i, ...] = \
                np.array(self['next_semantic_state'][_index], dtype=np.float32)

        states = \
            torch.ByteTensor(states).to(self.device).float()
        next_states = \
            torch.ByteTensor(next_states).to(self.device).float()
        semantic_states = torch.FloatTensor(semantic_states).to(self.device)
        semantic_next_states = torch.FloatTensor(semantic_next_states).to(self.device)
        actions = torch.FloatTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)
        collisions = torch.FloatTensor(self['collision'][indices]).to(self.device)
        ncol_values = torch.FloatTensor(self['factored_value'][indices, 0]).unsqueeze(1).to(self.device)
        col_values = torch.FloatTensor(self['factored_value'][indices, 1]).unsqueeze(1).to(self.device)

        return states, semantic_states, actions, rewards, collisions, next_states, semantic_next_states, \
               ncol_values, col_values, dones

    def __len__(self):
        return len(self['state'])

    # def get(self):
    #     return dict(self)

    def get(self):
        state_dict = {key: self[key] for key in self.state_keys}
        np_dict = {key: self[key][:self._n] for key in self.np_keys}
        state_dict.update(**np_dict)
        return state_dict

    def load(self, memory):
        try:
            for key in self.state_keys:
                self[key].extend(memory[key])

            num_data = len(memory['state'])
            if 'priority' in memory.keys():
                assert len(memory['state']) == len(memory['priority'])
            # for key in self.np_keys:
            #     print_flush('key {}, data len {}'.format(key, len(memory[key])))
            if self._p + num_data <= self.capacity:
                for key in self.np_keys:
                    self[key][self._p:self._p+num_data] = memory[key]
            else:
                mid_index = self.capacity - self._p
                end_index = num_data - mid_index
                for key in self.np_keys:
                    self[key][self._p:] = memory[key][:mid_index]
                    self[key][:end_index] = memory[key][mid_index:]

            self._n = min(self._n + num_data, self.capacity)
            self._p = (self._p + num_data) % self.capacity
            self.truncate()
            assert self._n == len(self)
        except Exception as e:
            error_handler(e)


class DummyMultiStepMemory(DummyMemory):

    def __init__(self, capacity, state_shape, semantic_state_shape, action_shape, device,
                 gamma=0.99, multi_step=3):
        super(DummyMultiStepMemory, self).__init__(
            capacity, state_shape, semantic_state_shape, action_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)
        self.trajectory_morphed_collision = []

    def append(self, state, semantic_state, action, reward, collision, next_state, next_semantic_state, factored_value,
               done, episode_done=False):
        try:
            if self.multi_step != 1:
                self.buff.append(state, semantic_state, action, reward)

                if len(self.buff) == self.multi_step:
                    state, semantic_state, action, reward = self.buff.get(self.gamma)
                    self._append(state, semantic_state, action, reward, collision, next_state, next_semantic_state,
                                 factored_value, done)

                if episode_done or done:
                    self.buff.reset()
                    # self.morph_collisions()
            else:
                self._append(state, semantic_state, action, reward, collision, next_state, next_semantic_state,
                             factored_value, done)
                # if episode_done or done:
                #     self.morph_collisions()
        except Exception as e:
            error_handler(e)

    def morph_collisions(self):
        pass
        # eps_len = len(self.trajectory_morphed_collision)
        # scane_range = 12
        # for i in range(eps_len):
        #     for j in range(i, min(i + scane_range, eps_len)):
        #         reverse_pos = j - eps_len
        #         if self['collision'][self._p + reverse_pos] > 1e-5:  # True
        #             self.trajectory_morphed_collision[i] = 1.0 * pow(self.gamma, j - i)
        #             break
        # self.revise_latest(flag='collision', new_data=self.trajectory_morphed_collision)
        # self.trajectory_morphed_collision.clear()

    def clear_multi_step_buff(self):
        if self.multi_step != 1:
            self.buff.reset()


class DummyPrioritizedMemory(DummyMultiStepMemory):
    state_keys = ['state', 'semantic_state', 'next_state', 'next_semantic_state']
    np_keys = ['action', 'reward', 'done', 'collision', 'priority', 'factored_value']
    keys = state_keys + np_keys

    def __init__(self, capacity, state_shape, semantic_state_shape, action_shape, device, gamma=0.99,
                 multi_step=3, alpha=0.6, beta=0.4, beta_annealing=0.001,
                 epsilon=1e-4):
        super(DummyPrioritizedMemory, self).__init__(
            capacity, state_shape, semantic_state_shape, action_shape, device, gamma, multi_step)
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def reset(self):
        super(DummyPrioritizedMemory, self).reset()
        self['priority'] = np.empty((self.capacity, 1), dtype=np.float32)

    def append(self, state, semantic_state, action, reward, collision, next_state, next_semantic_state,
               factored_value, done, error, episode_done=False):
        try:
            if self.multi_step != 1:
                self.buff.append(state, semantic_state, action, reward)

                if len(self.buff) == self.multi_step:
                    state, semantic_state, action, reward = self.buff.get(self.gamma)
                    self['priority'][self._p] = self.calc_priority(error)
                    self._append(state, semantic_state, action, reward, collision, next_state, next_semantic_state,
                                 factored_value, done)

                if episode_done or done:
                    self.buff.reset()
                    self.morph_collisions()
            else:
                self['priority'][self._p] = self.calc_priority(error)
                self._append(
                    state, semantic_state, action, reward, collision, next_state, next_semantic_state, factored_value, done)
                if episode_done or done:
                    self.morph_collisions()
        except Exception as e:
            error_handler(e)

    def update_priority(self, indices, errors):
        self['priority'][indices] = np.reshape(
            self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def get(self):
        try:
            state_dict = {key: self[key] for key in self.state_keys}
            np_dict = {key: self[key][:self._n] for key in self.np_keys}
            state_dict.update(**np_dict)
            return state_dict
        except Exception as e:
            error_handler(e)

    def sample(self, batch_size):
        try:
            self.beta = min(1. - self.epsilon, self.beta + self.beta_annealing)
            sampler = WeightedRandomSampler(
                self['priority'][:self._n, 0], batch_size)
            indices = list(sampler)

            batch = self._sample(indices, batch_size)
            priorities = np.array(self['priority'][indices], dtype=np.float32)
            priorities = priorities / np.sum(self['priority'][:self._n])

            weights = (self._n * priorities) ** -self.beta
            weights /= np.max(weights)
            weights = torch.FloatTensor(
                weights).view(batch_size, -1).to(self.device)

            return batch, indices, weights
        except Exception as e:
            error_handler(e)