import Pyro4
import random
import numpy as np
import cv2
from pathlib import Path
import os
import time
import sys

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
# sys.path.append(str(ws_root / 'il_controller' / 'src'))
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from utils import data_host, labeller_port, print_flush

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pos',
                        type=int,
                        default=0,
                        help='pos of data point')
    config = parser.parse_args()

    Pyro4.config.SERIALIZER = 'pickle'
    labeller_service = Pyro4.Proxy('PYRO:labellerservice.warehouse@{}:{}'.format(data_host, labeller_port))

    data = labeller_service.fetch_belief('inspector')
    for path_set in data.id_path_map:
        print_flush('agent {} has {} paths:'.format(path_set.agent_id, len(path_set.pathlist)))
        for path in path_set.pathlist:
            print_flush('l={}, m={}'.format(len(path.points), len(path.points) * 0.1))
