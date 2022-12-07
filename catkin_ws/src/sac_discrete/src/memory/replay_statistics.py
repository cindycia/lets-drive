#!/usr/bin/env python3

import os
import pickle
import sys
from collections import deque
from pathlib import Path
import numpy as np

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
sys.path.append(str(ws_root / 'il_controller' / 'src'))
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from utils import print_flush, replay_port, data_host, error_handler

SAVE_PATH = Path.home() / 'replay'
if not os.path.isdir(str(SAVE_PATH)):
    os.makedirs(str(SAVE_PATH))
# CAPACITY = 100000  # 500000
DECACHE_INTERVAL = 50
NORMALIZE_REWARD = False


def write_memory(key, memory):
    (SAVE_PATH / key).mkdir(parents=True, exist_ok=True)

    pickle.dump(memory, (SAVE_PATH / key / 'data').open('wb'))


def read_memory(key):
    memory = pickle.load((SAVE_PATH / key / 'data').open('rb'))
    return memory


if __name__ == "__main__":
    print_flush('[replay_statistics.py] ' + 'Replay service running.')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',
                        type=int,
                        default=0,
                        required=True,
                        help='replay memory size')
    parser.add_argument('--folder',
                        type=str,
                        default='~/replay',
                        required=True,
                        help='replay folder')
    args = parser.parse_args()

    # global CAPACITY
    CAPACITY = args.size
    capacity = CAPACITY
    print_flush('capacity={}'.format(capacity))
    multi_buffer = deque()
    buffer_size = 0
    num_blocks = 0

    try:
        existing_blocks = sorted([str(p).rstrip('/') for p in (SAVE_PATH).glob('*/')])
        for key in reversed(existing_blocks):
            if buffer_size < capacity:
                memory = read_memory(key)
                multi_buffer.appendleft(memory)
                buffer_size += len(memory['state'])
                num_blocks += 1
            else:
                break

        rewards = []
        for memory in multi_buffer:
            reward_list = memory['reward']
            accum_reward = np.sum(reward_list)
            rewards.append(accum_reward)

        rewards = np.asarray(rewards)
        print_flush('Ave_reward: {}'.format(np.mean(rewards)))
    except Exception as e:
        error_handler(e)
    total_num_data = buffer_size

    print_flush('[replay_service.py] ' + 'Initial buffer = {}'.format(buffer_size))
