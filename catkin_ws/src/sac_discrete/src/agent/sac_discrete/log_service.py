#!/usr/bin/env python3

import sys
import os

from pathlib import Path
import Pyro4
from datetime import datetime

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent.parent
sys.path.append(str(ws_root/'il_controller'/'src'))
sys.path.append(str(ws_root/'sac_discrete'/'src'))
from utils import print_flush, data_host, log_port

from torch.utils.tensorboard import SummaryWriter


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class LogService():
    def __init__(self):
        # self.writer = SummaryWriter('runs/{}'.format(SUB_FOLDER))
        self.writer = {}
        self.global_step = {}

    def add_log(self, num_steps, dir_flag, logs):
        if dir_flag not in self.global_step.keys():
            self.global_step[dir_flag] = {}
            self.writer[dir_flag] = SummaryWriter('runs/{}'.format(dir_flag))

        # print_flush('logging for the {} learner, global_step {}'.format(dir_flag, self.global_step[dir_flag]))

        for key in logs.keys():
            if key not in self.global_step[dir_flag].keys():
                self.global_step[dir_flag][key] = 0
            self.global_step[dir_flag][key] = self.global_step[dir_flag][key] + num_steps

            now = datetime.now()
            timestamp = datetime.timestamp(now)
            self.writer[dir_flag].add_scalars(key,
                                              {dir_flag: logs[key]},
                                              timestamp)
                                              # self.global_step[dir_flag][key])


def main():
    print_flush('[log_service.py] ' + 'Logging service running.')
    Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    Pyro4.Daemon.serveSimple(
            {
                LogService: "logservice.warehouse"
            },
            host=data_host,
            port=log_port,
            ns=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SUB_FOLDER = str(sys.argv[1])
    else:
        SUB_FOLDER = 'sac_losses'

    main()