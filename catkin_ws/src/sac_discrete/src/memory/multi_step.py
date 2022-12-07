from collections import deque
import numpy as np
from .base import Memory


class MultiStepBuff:
    keys = ["state", "semantic_state", "action", "reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.memory = {
            key: deque(maxlen=self.maxlen)
            for key in self.keys
            }

    def append(self, state, semantic_state, action, reward):
        self.memory["state"].append(state)
        self.memory["semantic_state"].append(semantic_state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        state = self.memory["state"].popleft()
        semantic_state = self.memory["semantic_state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        return state, semantic_state, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([
            r * (gamma ** i) for i, r
            in enumerate(self.memory["reward"])])

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in MultiStepBuff.')
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory['state'])


class MultiStepMemory(Memory):

    def __init__(self, capacity, state_shape, semantic_state_shape, action_shape, device,
                 gamma=0.99, multi_step=3):
        super(MultiStepMemory, self).__init__(
            capacity, state_shape, semantic_state_shape, action_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, semantic_state, action, reward, next_state, next_semantic_state, done,
               episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, semantic_state, action, reward)

            if len(self.buff) == self.multi_step:
                state, semantic_state, action, reward = self.buff.get(self.gamma)
                self._append(state, semantic_state, action, reward, next_state, next_semantic_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self._append(state, semantic_state, action, reward, next_state, next_semantic_state, done)
