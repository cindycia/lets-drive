#!/usr/bin/env python3

import collections
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

import Pyro4
import numpy as np

from utils import print_flush, data_host, env_port, labeller_port, error_handler

# ws_root = Path(os.path.realpath(__file__)).parent.parent.parent
# sys.path.append(str(ws_root / 'il_controller' / 'src'))
# sys.path.append(str(ws_root / 'reinforcement' / 'scripts'))

import rospy
from msg_builder.msg import Belief
from std_srvs.srv import Empty, EmptyResponse


class LabellerProxy(object):
    def __init__(self, port, mode='actor'):
        Pyro4.config.SERIALIZER = 'pickle'
        print_flush('[labeling_proxy] ' + 'Connecting to labeller service at PYRO:labellerservice.warehouse@{}:{}'.format(
            data_host, env_port + port))
        self.labeller_service = Pyro4.Proxy('PYRO:labellerservice.warehouse@{}:{}'.format(data_host, labeller_port))
        # self.labeller_service._pyroAsync()
        self.port = port
        if mode == 'actor':
            print_flush('[labeling_proxy] subscribing to /unlabelled_belief topic from the planner')
            rospy.Subscriber("unlabelled_belief", Belief, self.unlabelled_belief_call_back)
        elif mode == 'labeller':
            print_flush('[labeling_proxy] publishing /unlabelled_belief topic to the planner')
            self.belief_publisher = rospy.Publisher('unlabelled_belief', Belief, queue_size=1)
            self.fetch_data_service = rospy.Service('/fetch_data', Empty,
                                                           self.fetch_and_send_once)

    def fetch_and_send_once(self, req):
        try:
            data = None
            trial = 0
            while data is None and trial < 10:
                data = self.labeller_service.fetch_belief(self.port)
                if data is not None:
                    print_flush('[labeling_proxy] get data from labeller service')
                    for agent in data.agent_paths.agents:
                        agent.cross_dirs = list(agent.cross_dirs)
                        # print_flush('[labeling_proxy] id {}, type {}, {}'.format(agent.id, agent.type, agent.cross_dirs))
                    # print_flush('[labeling_proxy] data.agent_paths.agents={}'.format(data.agent_paths.agents))
                    self.belief_publisher.publish(data)
                else:
                    # print_flush('[labeling_proxy] no data from labeller service')
                    time.sleep(0.05)
                    trial += 1
            return EmptyResponse()
        except Exception as e:
            print_flush('[labeling_proxy.py] exception {}'.format(e))
            error_handler(e)

    def unlabelled_belief_call_back(self, belief_msg):
        try:
            print_flush('[labeling_proxy] ' + 'sending data to labeller_service')
            self.labeller_service.add_belief(belief_msg)
            print_flush('[labeling_proxy] ' + 'data sent to labeller_service')
        except Exception as e:
            print_flush('[labeling_proxy.py] exception {}'.format(e))
            error_handler(e)


if __name__ == '__main__':
    rospy.init_node('labeling_proxy')
    port = rospy.get_param('~port')
    mode = rospy.get_param('~mode')
    labeling_proxy = LabellerProxy(port, mode)
    rospy.spin()
