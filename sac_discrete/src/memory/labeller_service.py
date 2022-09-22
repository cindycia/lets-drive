#!/usr/bin/env python3

import os
import sys
import copy
import random
from collections import deque
from pathlib import Path
from threading import RLock
import Pyro4
from datetime import datetime


ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
# sys.path.append(str(ws_root / 'il_controller' / 'src'))
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from utils import print_flush, labeller_port, data_host, error_handler

CAPACITY = 10000  # 500000


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class LabellerService():
    def __init__(self):
        self.capacity = CAPACITY
        self.buffer_lock = RLock()
        self.buffer = deque()
        self.buffer_size = 0
        print_flush('[labeller_service.py] ' + 'Initial buffer = {}'.format(self.buffer_size))

    def add_belief(self, belief_msg):
        try:
            print_flush('[labeller_service.py] add_memory at {}'.format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

            self.buffer_lock.acquire()
            # forget old memory
            while self.buffer_size >= self.capacity:
                self.buffer.popleft()
                self.buffer_size -= 1
            # append new memory
            self.buffer.append(belief_msg)
            self.buffer_size += 1
            self.buffer_lock.release()

            print_flush('[labeller_service.py] ' + 'Buffer = {}'.format(self.buffer_size))
        except Exception as e:
            error_handler(e)

    def fetch_belief(self, label):
        try:
            if self.buffer_size > 1:  # has new memory
                self.buffer_lock.acquire()
                random_pos = random.randint(0, self.buffer_size - 1)
                print_flush('label {} fetching memory at pos {}'.format(label, random_pos))
                data = copy.deepcopy(self.buffer[random_pos])
                del self.buffer[random_pos]
                self.buffer_size -= 1
                print_flush('[labeller_service.py] ' + 'Buffer = {}'.format(self.buffer_size))
                self.buffer_lock.release()
                return data
            else:
                return None
        except Exception as e:
            error_handler(e)


if __name__ == "__main__":
    print_flush('[labeller_service.py] ' + 'Labeller service running.')
    Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    Pyro4.Daemon.serveSimple(
        {
            LabellerService: "labellerservice.warehouse"
        },
        host=data_host,
        port=labeller_port,
        ns=False)
