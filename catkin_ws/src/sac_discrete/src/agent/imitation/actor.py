import os
import sys
from datetime import datetime
from pathlib import Path
import time

import Pyro4
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent.parent
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from agent.imitation.base import ImitationAgent
from memory import DummyMultiStepMemory, DummyPrioritizedMemory
from policy import ConvCategoricalPolicy
from value import ConvVNetwork
from agent import hard_update
from env import CrowdDriving
from env.planner_reward import reward as reward_func
from utils import print_flush, log_flush, data_host, replay_port, error_handler_with_log, log_port, error_handler

LOAD_PATH = ws_root / 'sac_discrete' / 'trained_models'
SAVE_PATH = ws_root / 'crowd_pomdp_planner'


class ImitationActor(ImitationAgent):
    space_size = 65

    def __init__(self, env, log_dir, actor_id, true_actor, memory_size=1e4, gamma=0.99,
                 per=True, alpha=0.6, beta=0.4, beta_annealing=0.001,
                 multi_step=3, start_steps=10000, log_interval=10,
                 memory_save_interval=5, model_load_interval=5, cuda=True,
                 seed=0, batch_size=64, lr=0.0003, termination_step=0, offline_training=False):
        self.actor_id = actor_id
        self.true_actor = true_actor
        self.log_txt = open("actor_log_{}.txt".format(self.actor_id), "w")

        log_flush(self.log_txt, "[actor.py] init")
        self.env = env
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        log_flush(self.log_txt, '[actor.py] device={}'.format(self.device))

        self.policy = ConvCategoricalPolicy(
            self.env.observation_space.shape[0],
            self.env.semantic_observation_space.n,
            self.env.action_space.n).to(self.device).eval()
        self.value = ConvVNetwork(
            self.env.observation_space.shape[0],
            self.env.semantic_observation_space.n).to(self.device)
        self.value_target = ConvVNetwork(
            self.env.observation_space.shape[0],
            self.env.semantic_observation_space.n).to(self.device).eval()
        self.critic = None
        hard_update(self.value_target, self.value)

        if per:
            self.memory = DummyPrioritizedMemory(
                memory_size, self.env.observation_space.shape, self.env.semantic_observation_space.n,
                (1,), self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            self.memory = DummyMultiStepMemory(
                memory_size, self.env.observation_space.shape, self.env.semantic_observation_space.n,
                (1,), self.device, gamma, multi_step)

        Pyro4.config.COMMTIMEOUT = 0.0  # infinite wait
        Pyro4.config.SERIALIZER = 'pickle'
        log_flush(self.log_txt, '[actor.py] ' + 'Connecting to logging service...')
        self.logging_service = Pyro4.Proxy('PYRO:logservice.warehouse@{}:{}'.format(data_host, log_port))
        self.log_flag = 'lr_{}_bs_{}_ms_{}_seed_{}'.format(lr, batch_size, memory_size, seed)

        log_flush(self.log_txt, '[actor.py] ' + 'Connecting to replay service...')
        self.replay_service = Pyro4.Proxy('PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))
        self.virtual_replay_service = None
        # self.replay_service._pyroAsync()
        self.last_replay_block = 0

        self.episodes = 0
        self.steps = 0
        self.per = per
        self.multi_step = multi_step
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.log_interval = log_interval
        self.memory_save_interval = memory_save_interval
        self.model_load_interval = model_load_interval
        self.termination_step = termination_step

        if offline_training:
            self.start_steps = self.termination_step
            self.termination_step = self.termination_step * 2
        self.offline_training = offline_training

        self.episode_rewards = []
        self.step_vels = []
        self.collisions = []

        self.load_weights()
    
    def __del__(self):
        self.log_txt.close()

    def run(self):
        self.time = time.time()
        while self.total_data_count() <= self.termination_step:
            # if self.offline_training and self.total_data_count() > self.start_steps:
            #     break
            self.episodes += 1
            self.act_episode()
            self.interval()

    def act_episode(self):
        try:
            log_flush(self.log_txt, "[actor.py] act_episode")
            episode_steps = 0
            episode_done = False
            state, semantic_state = self.env.reset(False, os.path.exists(str(LOAD_PATH / '0' / 'policy')))

            if state is None:
                log_flush(self.log_txt, 'Environment reset failed. Wasting episode')
                return

            self.episode_rewards.append(0.)

            while not episode_done:
                self.env.render()
                # action = self.act(state) # no need to send action, I am using off-policy mode
                next_state, next_semantic_state, reward, collision,\
                terminal, offpolicy_action, variables = self.env.step(None)
                if terminal is None:  # time out without terminal
                    episode_done = True
                else:
                    episode_done = terminal

                self.steps += 1
                episode_steps += 1
                self.episode_rewards[-1] += reward
                if variables is not None:
                    self.step_vels.append(variables['vel'])
                    self.collisions.append(variables['col'])
                    factored_value = [variables['value_ncol'], variables['value_col']]
                else:
                    factored_value = None

                self.update_memory(state, next_state, offpolicy_action, semantic_state, next_semantic_state,
                                   factored_value, terminal, reward, collision, episode_done, self.true_actor)

                state = next_state
                semantic_state = next_semantic_state

            now = time.time()
            print(' ' * self.space_size,
                  f'Actor {self.actor_id:<2}  '
                  f'episode: {self.episodes:<4}  '
                  f'episode steps: {episode_steps:<4}  '
                  f'reward: {self.episode_rewards[-1]:<5.1f}  '
                  f'time: {now - self.time:3.3f}')
            self.time = now

            actor_alive = int(episode_steps > 1) + \
                          int(os.path.exists(str(LOAD_PATH / '0' / 'use_value'))) + \
                          os.path.exists(str(LOAD_PATH / '0' / 'policy'))
            if self.episodes % self.log_interval == 0:
                if self.true_actor:
                    log_dict = {
                        'reward/train': np.mean(np.asarray(self.episode_rewards)),
                        'reward/vel_train': np.mean(np.asarray(self.step_vels)),
                        'reward/collision_train': np.mean(np.asarray(self.collisions)),
                        'actor_alive/{}'.format(self.actor_id): actor_alive
                    }
                else:
                    log_dict = {
                        'actor_alive/{}'.format(self.actor_id): actor_alive
                    }
                self.episode_rewards.clear()
                self.step_vels.clear()
                self.collisions.clear()
                self.logging_service.add_log(1, self.log_flag, log_dict)

        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def interval(self):
        try:
            if self.episodes % self.model_load_interval == 0:
                self.load_weights()
            if self.episodes % self.memory_save_interval == 0:
                self.save_memory(self.actor_id)
        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def load_weights(self):
        log_flush(self.log_txt, "[actor.py] load_weights from {}".format(str(LOAD_PATH / '0' )))

        try:
            learner_id = '0'
            example_forward_input = (torch.rand(1, *self.env.observation_space.shape).to(self.device),
                                     torch.rand(1, self.env.semantic_observation_space.n).to(self.device))

            if os.path.exists(str(LOAD_PATH / learner_id / 'policy')):
                self.policy.load_state_dict(torch.load(str(LOAD_PATH / learner_id / 'policy')))
                self.value.load_state_dict(torch.load(str(LOAD_PATH / learner_id / 'value')))
                # self.value_target.load_state_dict(torch.load(str(LOAD_PATH / learner_id / 'value_target')))

                filename = str(ws_root / 'crowd_pomdp_planner' / 'temp_policy_net_{}.pt'.format(learner_id))
                log_flush(self.log_txt, '[actor.py] export ts policy to {}'.format(filename))
                ts_policy = torch.jit.trace(self.policy, example_forward_input)
                torch.jit.save(ts_policy, filename)
                if os.path.exists(str(LOAD_PATH / learner_id / 'use_value')):
                    filename = str(ws_root / 'crowd_pomdp_planner' / 'temp_value_net_{}.pt'.format(learner_id))
                    log_flush(self.log_txt, '[actor.py] export ts value to {}'.format(filename))
                    ts_value = torch.jit.trace(self.value, example_forward_input)
                    torch.jit.save(ts_value, filename)

                return True
            else:
                return False
        except Exception as e:
            error_handler(e)
            return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--aid',
                        type=int,
                        default=0,
                        help='GPU to use')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0003,
                        help='learning rate')
    parser.add_argument('--port',
                        type=int,
                        default=2000,
                        help='summit_port')
    parser.add_argument('--drive_mode',
                        type=str,
                        default="",
                        help='Which drive_mode to run')
    parser.add_argument('--env_mode',
                        type=str,
                        default="server",
                        help='display or server')
    parser.add_argument('--record',
                        type=bool,
                        default=False,
                        help='record rosbag')
    parser.add_argument('--offline',
                        type=bool,
                        default=False,
                        help='offline training mode')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--term', type=int, default=300000)
    args = parser.parse_args()
    true_actor = True
    if 'labeller' in args.drive_mode:
        true_actor = False

    import subprocess

    with subprocess.Popen('python3 {} --port {}'.format(
            str(ws_root / 'sac_discrete' / 'src' / 'env' / 'env_service.py'), args.port).split()) as env_server:
        with CrowdDriving(summit_port=args.port, gpu_id=args.aid, launch_env=args.env_mode,
                          drive_mode=args.drive_mode, record_bag=args.record, reward_func=reward_func) as driving_env:
            config = {
                'env': driving_env,
                'log_dir': os.path.join('logs', 'CrowdDriving',
                                        f'imitation-{datetime.now().strftime("%Y%m%d-%H%M")}'),
                'gamma': 0.95,
                'multi_step': 3,
                'per': True,  # prioritized experience replay
                'alpha': 0.6,
                'beta': 0.4,
                'beta_annealing': 0.001,
                'cuda': True,
                'seed': args.seed,
                'batch_size': 64,
                'lr': args.lr,
                'memory_size': 1e4,
                'log_interval': 1,
                'memory_save_interval': 1,
                'model_load_interval': 1,
                'actor_id': args.aid,
                'true_actor': true_actor,
                'start_steps': 10000,
                #'termination_step': 300000,
                'termination_step': args.term,
                'offline_training': args.offline
            }

            actor = ImitationActor(**config)
            actor.run()

    print_flush('[actor.py] termination')
