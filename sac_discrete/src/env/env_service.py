from pathlib import Path
import os
import sys
import Pyro4

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
sys.path.append(str(ws_root/'sac_discrete'/'src'))

from threading import RLock
from collections import deque
from utils import print_flush, data_host, env_port, error_handler
from env_proxy import ReplayData
import time

@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class EnvService():
    def __init__(self):
        self.buffer_lock = RLock()
        self.buffer = deque()
        self.initialized = False

    def add_data_point(self, key, transition):
        # print_flush('[env_service.py] env_proxy -> add_data_point, key={}'.format(key))
        # to be called by the lets-drive-zero planner
        self.buffer_lock.acquire()
        self.buffer.append((key, transition))
        self.buffer_lock.release()

    def pop_data_point(self):
        # to be called by the crowd-driving environment
        if len(self.buffer) == 0:
            return self.initialized, None, None

        # print_flush('[env_service.py] crowd_driving_env.step -> pop_data_point')
        self.initialized = True
        # data available
        self.buffer_lock.acquire()
        key, sample = self.buffer.popleft()
        self.buffer_lock.release()
        return self.initialized, key, sample

    def reset(self):
        # print_flush('[env_service.py] reset')
        # to be called by the crowd-driving environment
        self.buffer.clear()
        self.initialized = False


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--port',
                            type=int,
                            default=2000,
                            help='GPU to use')
        config = parser.parse_args()

        print_flush('[crowd_driving_env.py] ' + 'Env service running at port {}.'.format(env_port + config.port))
        Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
        Pyro4.Daemon.serveSimple(
            {
                EnvService: "envservice.warehouse"
            },
            host=data_host,
            port=env_port + config.port,
            ns=False)
    except Exception as e:
        error_handler(e)
