import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler

from .multi_step import MultiStepMemory


class PrioritizedMemory(MultiStepMemory):

    def __init__(self, capacity, state_shape, action_shape, device,
                 gamma=0.99, multi_step=3, alpha=0.6, beta=0.4,
                 beta_annealing=0.001, epsilon=1e-4):
        super(PrioritizedMemory, self).__init__(
            capacity, state_shape, action_shape, device, gamma, multi_step)
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def append(self, state, action, reward, next_state, done, error,
               episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if len(self.buff) == self.multi_step:
                state, action, reward = self.buff.get(self.gamma)
                self.priorities[self._p] = self.calc_priority(error)
                self._append(state, action, reward, next_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self.priorities[self._p] = self.calc_priority(error)
            self._append(state, action, reward, next_state, done)

    def update_priority(self, indices, errors):
        self.priorities[indices] = np.reshape(
            self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        self.beta = min(1. - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(
            self.priorities[:self._n, 0], batch_size)
        indices = list(sampler)
        batch = self._sample(indices)

        p = self.priorities[indices] / np.sum(self.priorities[:self._n])
        weights = (self._n * p) ** -self.beta
        weights /= np.max(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        return batch, indices, weights

    def reset(self):
        super(PrioritizedMemory, self).reset()
        self.priorities = np.empty(
            (self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid], self.actions[valid], self.rewards[valid],
            self.next_states[valid], self.dones[valid], self.priorities[valid])

    def _insert(self, mem_indices, batch, batch_indices):
        states, actions, rewards, next_states, dones, priorities = batch
        self.states[mem_indices] = states[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
        self.priorities[mem_indices] = priorities[batch_indices]
