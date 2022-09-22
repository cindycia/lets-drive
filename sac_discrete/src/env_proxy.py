#!/usr/bin/env python3

import collections
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

import Pyro4
import numpy as np

from utils import print_flush, data_host, env_port, error_handler

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent
sys.path.append(str(ws_root / 'il_controller' / 'src'))
sys.path.append(str(ws_root / 'reinforcement' / 'scripts'))

import rospy
from Config.global_params import config
from msg_builder.srv import SendStateAction, SendStateActionResponse

ReplayData = collections.namedtuple(
    'ReplayData',
    [
        'action',
        'next_state',
        'next_state_semantic',
        'next_variables'
    ])


class EnvProxy(object):
    def __init__(self, port):
        Pyro4.config.SERIALIZER = 'pickle'
        print_flush('[env_proxy] ' + 'Connecting to env service at PYRO:envservice.warehouse@{}:{}'.format(
            data_host, env_port + port))
        self.env_service = Pyro4.Proxy('PYRO:envservice.warehouse@{}:{}'.format(data_host, env_port + port))
        # self.env_service._pyroAsync()

        print_flush('[env_proxy] starting /send_state_action service')
        self.send_state_action_service = rospy.Service('/send_state_action', SendStateAction, self.send_state_action)
        self.previous_action = None

    def send_state_action(self, req):
        try:
            print_flush('[env_proxy] ' + 'Get state action pair from planner...')

            state = np.frombuffer(req.state, dtype=np.uint8).reshape(
                (config.total_num_channels, config.imsize, config.imsize))
            semantic = list(req.semantic)

            collision = 0
            if float(req.reward) < -100:
                collision = 1

            variables = {'reward': float(req.reward),
                         'is_terminal': int(req.is_terminal),
                         'vel': float(req.vel),
                         'ttc': float(req.ttc),
                         'col': float(collision),
                         'value_col': float(req.value_col_factor),
                         'value_ncol': float(req.value_ncol_factor)}
            # action_probs = req.action_probs
            # action_value = req.value
            action = req.action

            if req.is_terminal:
                print_flush('[env_proxy] ' + 'get terminal state')

            transition = ReplayData(
                self.previous_action,
                state,
                semantic,
                variables)

            print_flush('[env_proxy] ' + 'sending data to env_service')
            self.env_service.add_data_point('{:.9f}'.format(Decimal(time.time())), transition)
            self.previous_action = action
            print_flush('[env_proxy] ' + 'data sent to env_service')

            response = SendStateActionResponse(True)
            return response
        except Exception as e:
            print_flush('[env_proxy.py] exception {}'.format(e))
            error_handler(e)



if __name__ == '__main__':
    rospy.init_node('env_proxy')
    port = rospy.get_param('~port')
    env_proxy = EnvProxy(port)
    rospy.spin()
