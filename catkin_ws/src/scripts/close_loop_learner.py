from multiprocessing import Process
import collections
import sys
from os.path import expanduser
import time

learner_scripts = expanduser('~/catkin_ws/src/reinforcement/scripts')

sys.path.append(learner_scripts)

import replay_service, parameters_service, sac_train


def print_flush(msg):
    print(msg)
    sys.stdout.flush()


class ReplayServer(Process):
    def __init__(self, config, ):
        Process.__init__(self)
        self.verbosity = config.verbosity

    def run(self):
        if self.verbosity > 0:
            print_flush("[close_loop_learner.py] launching replay_server; log: replay_log.txt")
        # sys.stdout = open('replay_log.txt', "w")

        replay_service.main()


class ParamServer(Process):
    def __init__(self, config, ):
        Process.__init__(self)
        self.verbosity = config.verbosity

    def run(self):
        if self.verbosity > 0:
            print_flush("[close_loop_learner.py] launching param_server; log: parameter_log.txt")
        # sys.stdout = open('parameter_log.txt', "w")

        parameters_service.main()


class Learner(Process):
    def __init__(self, config):
        Process.__init__(self)
        self.verbosity = config.verbosity
        self.gpu_id = config.gpu_id

    def run(self):
        if self.verbosity > 0:
            print_flush("[close_loop_learner.py] launching replay; log: learner_log.txt")
        # sys.stdout = open('learner_log.txt', "w")

        sac_train.main(self.gpu_id)


def main(config):
    import torch
    old_method = torch.multiprocessing.get_start_method()
    torch.multiprocessing.set_start_method('spawn', force=True)

    replay = ReplayServer(config)
    replay.start()
    param_s = ParamServer(config)
    param_s.start()

    time.sleep(1)

    learner = Learner(config)
    learner.start()

    torch.multiprocessing.set_start_method(old_method, force=True)

    return replay, param_s, learner


if __name__ == "__main__":
    Args = collections.namedtuple('args', 'verbosity gpu_id')

    # Spawn meshes.
    config = Args(
        verbosity=1,
        gpu_id=0)
    
    replay, param_s, learner = main(config)

    time.sleep(100.0)