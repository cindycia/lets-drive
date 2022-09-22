import numpy as np
import torch


class Memory:

    def __init__(self, capacity, state_shape, semantic_state_shape, action_shape, device):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.semantic_state_shape = semantic_state_shape
        self.action_shape = action_shape
        self.device = device
        self.is_image = len(state_shape) == 3
        self.state_type = np.uint8 if self.is_image else np.float32

        self.reset()

    def append(self, state, semantic_state, action, reward, next_state, next_semantic_state, done,
               episode_done=None):
        self._append(state, semantic_state, action, reward, next_state, next_semantic_state, done)

    def _append(self, state, semantic_state, action, reward, next_state, next_semantic_state, done):
        state = np.array(state, dtype=self.state_type)
        next_state = np.array(next_state, dtype=self.state_type)

        semantic_state = np.array(semantic_state, dtype=np.float32)
        next_semantic_state = np.array(next_semantic_state, dtype=np.float32)

        self.states[self._p] = state
        self.semantic_states[self._p] = semantic_state
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.next_states[self._p] = next_state
        self.next_semantic_states[self._p] = next_semantic_state
        self.dones[self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return self._sample(indices)

    def _sample(self, indices):
        if self.is_image:
            states = self.states[indices].astype(np.uint8)
            next_states = self.next_states[indices].astype(np.uint8)
            states = \
                torch.ByteTensor(states).to(self.device).float()
            next_states = \
                torch.ByteTensor(next_states).to(self.device).float()
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)

        semantic_states = self.semantic_states[indices]
        next_semantic_states = self.next_semantic_states[indices]
        semantic_states = torch.FloatTensor(semantic_states).to(self.device)
        next_semantic_states = torch.FloatTensor(next_semantic_states).to(self.device)

        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, semantic_states, actions, rewards, next_states, next_semantic_states, dones

    def __len__(self):
        return self._n

    def reset(self):
        self._n = 0
        self._p = 0

        self.states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type)
        self.semantic_states = np.empty(
            (self.capacity, *self.semantic_state_shape), dtype=np.float32)
        self.actions = np.empty(
            (self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty(
            (self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type)
        self.next_semantic_states = np.empty(
            (self.capacity, *self.semantic_state_shape), dtype=np.float32)
        self.dones = np.empty(
            (self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid], self.semantic_states[valid], self.actions[valid], self.rewards[valid],
            self.next_states[valid], self.next_semantic_states[valid], self.dones[valid])

    def load(self, batch):
        num_data = len(batch[0])

        if self._p + num_data <= self.capacity:
            self._insert(
                slice(self._p, self._p+num_data), batch,
                slice(0, num_data))
        else:
            mid_index = self.capacity-self._p
            end_index = num_data - mid_index
            self._insert(
                slice(self._p, self.capacity), batch,
                slice(0, mid_index))
            self._insert(
                slice(0, end_index), batch,
                slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, mem_indices, batch, batch_indices):
        states, semantic_states, actions, rewards, next_states, next_semantic_states, dones = batch
        self.states[mem_indices] = states[batch_indices]
        self.semantic_states[mem_indices] = semantic_states[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.next_semantic_states[mem_indices] = next_semantic_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
