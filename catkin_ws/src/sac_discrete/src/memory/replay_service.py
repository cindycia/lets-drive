#!/usr/bin/env python3

import os
import pickle
import sys
from collections import deque
from pathlib import Path
from threading import RLock
from time import time
import subprocess
import Pyro4
from datetime import datetime

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
sys.path.append(str(ws_root / 'il_controller' / 'src'))
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from utils import print_flush, replay_port, virtual_replay_port, data_host, error_handler

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


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class ReplayService():
    def __init__(self):
        self.capacity = CAPACITY
        print_flush('capacity={}'.format(self.capacity))
        self.buffer_lock = RLock()
        self.multi_buffer = deque()
        self.buffer_size = 0
        self.num_blocks = 0

        try:
            existing_blocks = sorted([str(p).rstrip('/') for p in (SAVE_PATH).glob('*/')])
            for key in reversed(existing_blocks):
                if self.buffer_size < self.capacity:
                    memory = read_memory(key)
                    self.multi_buffer.appendleft(memory)
                    self.buffer_size += len(memory['state'])
                    self.num_blocks += 1
                else:
                    break

            if not os.path.isdir(str(ws_root / 'sac_discrete' / 'logs')):
                os.makedirs(str(ws_root / 'sac_discrete' / 'logs'))
        except Exception as e:
            error_handler(e)
        self.total_num_data = self.buffer_size

        print_flush('[replay_service.py] ' + 'Initial buffer = {}'.format(self.buffer_size))

    def add_memory(self, key, memory, actor_id):
        try:
            print_flush('[replay_service.py] add_memory from actor {} at {}'.format(
                actor_id, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            write_memory(key, memory)
            with open(str(ws_root / 'sac_discrete' / 'logs' / 'replay.txt'), 'a') as f:
                f.write('{} {}\n'.format(key, len(memory)))

            self.buffer_lock.acquire()
            # forget old memory
            while self.buffer_size >= self.capacity:
                forgotten = self.multi_buffer.popleft()
                self.buffer_size -= len(forgotten['state'])
            # append new memory
            self.multi_buffer.append(memory)
            self.buffer_size += len(memory['state'])
            self.total_num_data += len(memory['state'])
            self.num_blocks += 1
            self.buffer_lock.release()

            print_flush('[replay_service.py] ' + 'Buffer = {}, All_blocks = {}, total_data={}'.format(
                self.buffer_size, self.num_blocks, self.total_num_data))
        except Exception as e:
            error_handler(e)

    def fetch_memory(self, last_block):
        try:
            if last_block < self.num_blocks:  # has new memory
                negative_pos = last_block - self.num_blocks
                print_flush('fetching memory at block {}'.format(negative_pos))
                return last_block + 1, self.multi_buffer[negative_pos]
            else:
                return last_block, None
        except Exception as e:
            error_handler(e)

    def total_data_count(self):
        return self.total_num_data


if __name__ == "__main__":
    print_flush('[replay_service.py] ' + 'Replay service running.')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',
                        type=int,
                        default=0,
                        required=True,
                        help='replay memory size')
    parser.add_argument('--type',
                        type=str,
                        default='real',
                        help='replay type: real / virtual')
    args = parser.parse_args()
    # global CAPACITY
    CAPACITY = args.size

    Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    if args.type == 'real':
        Pyro4.Daemon.serveSimple(
            {
                ReplayService: "replayservice.warehouse"
            },
            host=data_host,
            port=replay_port,
            ns=False)
    else:
        Pyro4.Daemon.serveSimple(
            {
                ReplayService: "virtualreplayservice.warehouse"
            },
            host=data_host,
            port=virtual_replay_port,
            ns=False)
