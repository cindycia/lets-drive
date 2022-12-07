import os
from time import time
import numpy as np
import torch
from torch.optim import Adam
from datetime import datetime
import pickle
from pathlib import Path
import Pyro4
import sys

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent.parent
print(ws_root)
sys.stdout.flush()
SAVE_PATH = ws_root / 'sac_discrete' / 'trained_models'
sys.path.append(str(ws_root/'sac_discrete'/'src'))

from agent.sac_discrete.base import SacDiscreteAgent
from memory import DummyMultiStepMemory, DummyPrioritizedMemory
from policy import ConvCategoricalPolicy
from q_function import TwinedDiscreteConvQNetwork
from value import ConvVNetwork
from agent import soft_update, hard_update, update_params
from env import CrowdDriving
from utils import data_host, log_port, replay_port, virtual_replay_port, print_flush, error_handler
from env.reward import explained_variance_score
from env.planner_reward import reward as reward_func


class SacDiscreteLearner(SacDiscreteAgent):

    def __init__(self, env, log_dir, learner_id=0,
                 batch_size=64, lr=0.0003, memory_size=1e5, use_onpolicy_data=False, gamma=0.99,
                 tau=0.005, multi_step=3, per=True, alpha=0.6, beta=0.4,
                 beta_annealing=0.001, grad_clip=5.0, update_per_steps=4, policy_pt_delay=0,
                 start_steps=1000, log_interval=1, memory_load_interval=5,
                 target_update_interval=1, model_save_interval=5, model_checkpoint_interval=100,
                 entropy_annealing=0.0, entropy_anneal_interval=1000,
                 eval_interval=1000, cuda=True, seed=0, termination_step=350000,
                 value_learning_delay=1000, value_pt_delay=1000):

        self.log_flag = 'lr_{}_bs_{}_ms_{}_seed_{}'.format(lr, batch_size, memory_size, seed)
        self.writer = None
        self.env = env
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        print_flush('device {}'.format(self.device))

        torch.autograd.set_detect_anomaly(True)

        self.policy = ConvCategoricalPolicy(
            self.env.observation_space.shape[0],
            self.env.semantic_observation_space.n,
            self.env.action_space.n).to(self.device)
        self.critic = TwinedDiscreteConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.semantic_observation_space.n,
            self.env.action_space.n).to(self.device)
        self.critic_target = TwinedDiscreteConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.semantic_observation_space.n,
            self.env.action_space.n).to(self.device).eval()
        self.value = ConvVNetwork(
            self.env.observation_space.shape[0],
            self.env.semantic_observation_space.n).to(self.device)

        hard_update(self.critic_target, self.critic)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)
        self.value_optim = Adam(self.value.parameters(), lr=lr)

        self.target_entropy = np.log(self.env.action_space.n) * 0.98
        self.min_target_entropy = np.log(self.env.action_space.n) * 0.92  # max_prob 0.2
        # self.min_target_entropy = np.log(self.env.action_space.n) * 0.65  # max_prob 0.5
        # self.min_target_entropy = np.log(self.env.action_space.n) * 0.35

        self.entropy_annealing = entropy_annealing
        self.entropy_anneal_interval = entropy_anneal_interval
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self.learner_id = str(learner_id)
        self.value_learning_delay = value_learning_delay
        self.value_pt_delay = value_pt_delay
        self.policy_pt_delay = policy_pt_delay

        self.steps = 0
        self.save_weights(leaner_id=learner_id)
        if per:
            self.memory = DummyPrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.semantic_observation_space.n,
                (1,), self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
            self.virtual_memory = DummyPrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.semantic_observation_space.n,
                (1,), self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            self.memory = DummyMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.semantic_observation_space.n,
                (1,), self.device, gamma, multi_step)
            self.virtual_memory = DummyMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.semantic_observation_space.n,
                (1,), self.device, gamma, multi_step)

        self.use_onpolicy_data = use_onpolicy_data

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        Pyro4.config.COMMTIMEOUT = 0.0  # infinite wait
        Pyro4.config.SERIALIZER = 'pickle'
        self.logging_service = Pyro4.Proxy('PYRO:logservice.warehouse@{}:{}'.format(data_host, log_port))

        Pyro4.config.SERIALIZER = 'pickle'
        print_flush('[learner.py] ' + 'Connecting to replay service at '
                                      'PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))
        self.replay_service = Pyro4.Proxy('PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))
        print_flush('[learner.py] ' + 'Connecting to virtual replay service at '
                                      'PYRO:virtualreplayservice.warehouse@{}:{}'.format(data_host, virtual_replay_port))
        self.virtual_replay_service = Pyro4.Proxy('PYRO:virtualreplayservice.warehouse@{}:{}'.format(
            data_host, virtual_replay_port))

        # self.replay_service._pyroAsync()
        self.last_replay_block = 0
        self.last_virtual_replay_block = 0

        self.epochs = 0
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.grad_clip = grad_clip
        self.update_per_steps = update_per_steps
        self.log_interval = log_interval
        self.memory_load_interval = memory_load_interval
        self.model_save_interval = model_save_interval
        self.model_checkpoint_interval = model_checkpoint_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.termination_step = termination_step

    def run(self):
        try:
            self.load_memory()
            print_flush('[learner] INFO: current memory size {} / {} / {}'.format(self.size(), len(self.memory), self.total_data_count()))

            self.time = time()
            while self.size() < self.start_steps:
                self.evaluate()
                self.load_memory()
                print_flush('[learner] INFO: current memory size {} / {} / {}'.format(self.size(), len(self.memory), self.total_data_count()))

            # self.time = time()
            while self.total_data_count() <= self.termination_step:
                print_flush('[learner] INFO: current memory size {} / {} / {}'.format(self.size(), len(self.memory), self.total_data_count()))
                self.epochs += 1
                for _ in range(self.update_per_steps):
                    self.steps += 1
                    self.learn()
                    self.interval()
        except Exception as e:
            error_handler(e)

    def learn(self):
        try:
            if self.steps % 1000 == 0:
                print_flush('step {}'.format(self.steps))
            if self.per:
                batch, indices, weights = \
                    self.memory.sample(self.batch_size)
                if len(self.virtual_memory) > 0:
                    vm_batch, vm_indices, vm_weights = \
                        self.virtual_memory.sample(self.batch_size)
                else:
                    print_flush('no virtual memory available')
                    vm_batch, vm_indices, vm_weights = \
                        batch, indices, weights
            else:
                batch = self.memory.sample(self.batch_size)
                weights = 1.
                if len(self.virtual_memory) > 0:
                    vm_batch = self.virtual_memory.sample(self.batch_size)
                    vm_weights = 1.
                else:
                    print_flush('no virtual memory available')
                    vm_batch, vm_weights = batch, weights

            q1_loss, q2_loss, errors, mean_q1, mean_q2, evar_1, evar_2 =\
                self.calc_critic_loss(batch, weights)
            policy_loss, entropies, pi = self.calc_policy_loss(batch, weights)

            if self.per:
                self.memory.update_priority(indices, errors.cpu().numpy())

            combined_loss, value_loss, col_loss, v_mask_loss, col_mask_loss, errors, \
            mean_v, mean_col, mean_target_v, mean_target_col, evar_v, evar_col, \
            mask_accuracy, mean_v_mask, mean_col_mask, mean_target_v_mask, mean_target_col_mask = \
                self.calc_value_loss(vm_batch, vm_weights)

            if self.per:
                if len(self.virtual_memory) > 0:
                    self.virtual_memory.update_priority(indices, errors.cpu().numpy())
                else:
                    self.memory.update_priority(indices, errors.cpu().numpy())

            entropy_loss = self.calc_entropy_loss(entropies, weights)

            update_params(
                self.policy_optim, self.policy, policy_loss, self.grad_clip, retain_graph=True)
            update_params(
                self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip, retain_graph=True)
            update_params(
                self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip, retain_graph=True)
            if self.steps >= self.value_learning_delay:
                update_params(
                    self.value_optim, self.value, combined_loss, self.grad_clip, retain_graph=True)

            self.update_params(
                self.alpha_optim, None, entropy_loss)

            self.alpha = self.log_alpha.exp()

            if self.steps % self.log_interval == 0:
                pi_max, _ = pi.detach().max(dim=1)
                states, semantic_states, _, _, _, _, _, _, _, _ = batch
                _, updated_pi, _, _ = \
                    self.policy.sample(states, semantic_states)
                ori_pi = torch.distributions.Categorical(pi.detach().cpu())
                updated_pi = torch.distributions.Categorical(updated_pi.detach().cpu())
                policy_kl = torch.distributions.kl.kl_divergence(ori_pi, updated_pi)
                policy_grad_norm = self.calc_grad_mag(self.policy)
                value_grad_norm = self.calc_grad_mag(self.value)
                if self.steps >= self.value_learning_delay:
                    noncol_value_loss = value_loss.detach().item()
                    col_value_loss = col_loss.detach().item()
                    noncol_mask_loss = v_mask_loss.detach().item()
                    col_mask_loss = col_mask_loss.detach().item()
                else:
                    noncol_value_loss = 1.0
                    col_value_loss = 1.0
                    noncol_mask_loss = 1.0
                    col_mask_loss = 1.0

                log_dict = {
                    'loss/value_ncol': noncol_value_loss,
                    'loss/value_col': col_value_loss,
                    'loss/mask_ncol': noncol_mask_loss,
                    'loss/mask_col': col_mask_loss,
                    'loss/mask_accuracy': mask_accuracy,
                    'loss/evar_value_ncol': evar_v,
                    'loss/evar_value_col': evar_col,
                    'loss/Q1': q1_loss.detach().item(),
                    'loss/Q2': q2_loss.detach().item(),
                    'loss/policy': policy_loss.detach().item(),
                    'loss/alpha': entropy_loss.detach().item(),
                    'stats/alpha': self.alpha.detach().item(),
                    'stats/mean_Q1': mean_q1,
                    'stats/mean_Q2': mean_q2,
                    'stats/evar_Q1': evar_1,
                    'stats/evar_Q2': evar_2,
                    'stats/entropy': entropies.detach().mean().item(),
                    'stats/target_entropy': self.target_entropy,
                    'stats/max_prob': pi_max.cpu().mean(),
                    'stats/policy_kl': policy_kl.mean(),
                    'stats/policy_grad_norm': policy_grad_norm,
                    'stats/value_grad_norm': value_grad_norm,
                    'stats/mean_value_ncol': mean_v,
                    'stats/mean_value_ncol_target': mean_target_v,
                    'stats/mean_value_col': mean_col,
                    'stats/mean_value_col_target': mean_target_col,
                    'stats/mean_mask_ncol': mean_v_mask,
                    'stats/mean_mask_ncol_target': mean_target_v_mask,
                    'stats/mean_mask_col': mean_col_mask,
                    'stats/mean_mask_col_target': mean_target_col_mask,
                    'progress/total_data_count': self.total_data_count()
                }
                self.logging_service.add_log(self.log_interval, self.log_flag, log_dict)
        except Exception as e:
            error_handler(e)

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        errors = torch.abs(curr_q1.detach() - target_q)
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        evar_1 = explained_variance_score(target_q.cpu(), curr_q1.detach().cpu())
        evar_2 = explained_variance_score(target_q.cpu(), curr_q2.detach().cpu())

        return q1_loss, q2_loss, errors, mean_q1, mean_q2, evar_1, evar_2

    def calc_policy_loss(self, batch, weights):
        try:
            states, semantic_states, actions, rewards, collisions, \
            next_states, semantic_next_states, ncol_values, col_values, dones = batch
            _, action_probs, log_action_probs, _ =\
                self.policy.sample(states, semantic_states)
            q1, q2 = self.critic(states, semantic_states)
            q = self.alpha * log_action_probs - torch.min(q1, q2)
            inside_term = torch.sum(action_probs * q, dim=1, keepdim=True)
            policy_loss = (inside_term * weights).mean()
            entropies = -torch.sum(
                action_probs * log_action_probs, dim=1, keepdim=True)
            return policy_loss, entropies, action_probs
        except Exception as e:
            error_handler(e)
            exit(-1)

    def calc_value_loss(self, batch, weights):
        try:
            weights = 1.0  # no weighted gradient, only biased sampling
            curr_v, curr_col, cur_v_mask, cur_col_mask, raw_v, raw_col = self.calc_current_v(*batch)
            target_v, target_col, target_v_mask, target_col_mask = self.calc_target_v(*batch)

            masked_v = raw_v * target_v_mask
            masked_col = raw_col * target_col_mask

            v_scale = 10.0
            v_loss = torch.mean((masked_v - target_v).pow(2) * weights)
            col_loss = torch.mean((masked_col - target_col).pow(2) * weights)
            # combined_loss = torch.max(v_loss, col_loss) / v_scale
            v_mask_loss = torch.mean((cur_v_mask - target_v_mask).pow(2) * weights)
            col_mask_loss = torch.mean((cur_col_mask - target_col_mask).pow(2) * weights)
            combined_mask_loss = torch.max(v_mask_loss, col_mask_loss)

            col_mask_accuacy = ((cur_col_mask >= 0.5).float() == target_col_mask).float().mean()
            v_mask_accuacy = ((cur_v_mask >= 0.5).float() == target_v_mask).float().mean()

            mask_accuracy = min(col_mask_accuacy, v_mask_accuacy)

            combined_loss = combined_mask_loss
            if col_mask_accuacy >= 0.95:
                combined_loss = combined_loss + col_loss / v_scale
            if v_mask_accuacy >= 0.95:
                combined_loss = combined_loss + v_loss / v_scale
            # errors = self.calc_v_error(curr_v, curr_col, target_v, target_col)
            errors = self.calc_mask_error(cur_v_mask, cur_col_mask, target_v_mask, target_col_mask)

            mean_v = curr_v.detach().mean().item()
            mean_target_v = target_v.mean().item()
            mean_col = curr_col.detach().mean().item()
            mean_target_col = target_col.mean().item()
            mean_v_mask = cur_v_mask.detach().mean().item()
            mean_col_mask = cur_col_mask.detach().mean().item()
            mean_target_v_mask = target_v_mask.mean().item()
            mean_target_col_mask = target_col_mask.mean().item()

            evar_v = explained_variance_score(target_v.cpu(), masked_v.detach().cpu())
            evar_col = explained_variance_score(target_col.cpu(), masked_col.detach().cpu())

            return combined_loss, v_loss, col_loss, v_mask_loss, col_mask_loss, errors, \
                   mean_v, mean_col, mean_target_v, mean_target_col, evar_v, evar_col, \
                   mask_accuracy, mean_v_mask, mean_col_mask, mean_target_v_mask, mean_target_col_mask
        except Exception as e:
            error_handler(e)

    def calc_entropy_loss(self, entropies, weights):
        entropy_loss = -(
            self.log_alpha * weights
            * (self.target_entropy-entropies).detach()
            ).mean()
        return entropy_loss

    def size(self):
        return len(self.memory) + len(self.virtual_memory)

    def interval(self):
        if self.steps % self.eval_interval == 0:
            self.evaluate()
            if self.total_data_count() > self.start_steps:
                self.eval_interval = min(int(self.size() * 0.1), 5000)
                print_flush('[learner] adjust eval_interval to {}'.format(self.eval_interval))
        if self.steps % self.memory_load_interval == 0:
            self.load_memory()
        if self.steps % self.model_save_interval == 0:
            self.save_weights(self.learner_id)
        if self.steps % self.model_checkpoint_interval == 0:
            self.save_models()
        if self.steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        if self.steps % self.entropy_anneal_interval == 0:
            if self.total_data_count() > self.start_steps:
                self.target_entropy = max(self.target_entropy - self.entropy_annealing, self.min_target_entropy)

    def evaluate(self):
        try:
            self.save_pt()

            episodes = 3
            returns = np.zeros((episodes,), dtype=np.float32)

            step_vels = []
            collisions = []
            effective_episodes = 0
            while effective_episodes < episodes:
                print_flush('eval episode {}'.format(effective_episodes))
                state, semantic_state = self.env.reset(True, os.path.exists(str(SAVE_PATH / '0' / 'policy')))

                if state is None:
                    print_flush('Environment reset failed. Wasting episode')
                    return

                episode_reward = 0.
                episode_done = False
                episode_length = 0
                while not episode_done:
                    self.env.render()
                    # action = self.exploit(state)
                    next_state, next_semantic_state, reward, collision, \
                    terminal, offpolicy_action, variables = self.env.step(None)

                    if terminal is None:  # time out without terminal
                        episode_done = True
                    else:
                        episode_done = terminal

                    # episode_reward += reward
                    if variables is not None:
                        step_vels.append(variables['vel'])
                        collisions.append(variables['col'])
                        factored_value = [variables['value_ncol'], variables['value_col']]
                        episode_reward += reward_func(offpolicy_action, variables['vel'], variables['ttc'],
                                                      variables['is_terminal'], variables['col'])
                    else:
                        factored_value = None
                    self.update_memory(state, next_state, offpolicy_action, semantic_state, next_semantic_state,
                        factored_value, terminal, reward, collision, episode_done, true_actor=True)

                    episode_length += 1
                    state = next_state
                    semantic_state = next_semantic_state

                if episode_length > 1:
                    returns[effective_episodes] = episode_reward
                    effective_episodes += 1

            mean_return = np.mean(returns)
            log_dict = {
                'reward/test': mean_return,
                'reward/vel_test': np.mean(np.asarray(step_vels)),
                'reward/collision_test': np.mean(np.asarray(collisions))
            }
            self.logging_service.add_log(episodes, self.log_flag, log_dict)

            now = time()
            print('Learner  '
                  f'Num steps: {self.steps:<5}  '
                  f'reward: {mean_return:<5.1f}  '
                  f'time: {now - self.time:3.3f}')
            self.time = now

        except Exception as e:
            error_handler(e)

    def save_models(self):
        torch.save(self.policy.state_dict(), str(SAVE_PATH / self.learner_id / 'policy_cp_{}'.format(self.steps)))
        torch.save(self.critic.state_dict(), str(SAVE_PATH / self.learner_id / 'critic_cp_{}'.format(self.steps)))
        torch.save(self.critic_target.state_dict(), str(SAVE_PATH / self.learner_id / 'critic_target_cp_{}'.format(self.steps)))

    def save_weights(self, leaner_id):
        try:
            # print_flush('[learner.py] save weights to {}'.format(str(SAVE_PATH / str(leaner_id))))
            leaner_id = str(leaner_id)
            # print_flush('SAVE_PATH={}'.format(SAVE_PATH))
            if not os.path.isdir(str(SAVE_PATH / leaner_id)):
                os.makedirs(str(SAVE_PATH / leaner_id))

            if self.steps >= self.policy_pt_delay:
                torch.save(self.policy.state_dict(), str(SAVE_PATH / leaner_id / 'policy'))
                torch.save(self.critic.state_dict(), str(SAVE_PATH / leaner_id / 'critic'))
                torch.save(self.critic_target.state_dict(), str(SAVE_PATH / leaner_id / 'critic_target'))
                torch.save(self.value.state_dict(), str(SAVE_PATH / leaner_id / 'value'))
                if self.steps >= self.value_learning_delay + self.value_pt_delay:
                    torch.save(torch.ones(1), str(SAVE_PATH / leaner_id / 'use_value'))

                pickle.dump(self.alpha.clone().detach().item(), (SAVE_PATH / leaner_id / 'alpha').open('wb'))

            # pickle.dump(deepcopy(self.policy).cpu().state_dict(), (SAVE_PATH / leaner_id / 'policy').open('wb'))
            # pickle.dump(deepcopy(self.critic).cpu().state_dict(), (SAVE_PATH / leaner_id / 'critic').open('wb'))
            # pickle.dump(deepcopy(self.critic_target).cpu().state_dict(), (SAVE_PATH / leaner_id / 'critic_target').open('wb'))
        except Exception as e:
            error_handler(e)

    def save_pt(self):
        try:
            example_forward_input = (torch.rand(2, *self.env.observation_space.shape).to(self.device),
                                     torch.rand(2, self.env.semantic_observation_space.n).to(self.device))

            if os.path.exists(str(SAVE_PATH / self.learner_id / 'policy')):
                filename = str(ws_root / 'crowd_pomdp_planner' / 'temp_policy_net_{}.pt'.format(self.learner_id))
                print_flush('[learner.py] export ts policy to {} for evaluation'.format(filename))
                ts_policy = torch.jit.trace(self.policy, example_forward_input)
                # ts_policy = ts_policy.cuda(self.device)
                torch.jit.save(ts_policy, filename)
            return True
        except Exception as e:
            print_flush(e)
            return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
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
                        default="lets-drive-zero",
                        help='Which drive_mode to run')
    parser.add_argument('--env_mode',
                        type=str,
                        default="server",
                        help='display or server')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    import subprocess
    with subprocess.Popen('python3 {} --port {}'.format(
        str(ws_root / 'sac_discrete' / 'src' / 'env' / 'env_service.py'), args.port).split()) as env_server:
        with CrowdDriving(summit_port=args.port, gpu_id=args.gpu, launch_env=args.env_mode,
                                drive_mode=args.drive_mode, record_bag=False) as driving_env:
            config = {
                'env': driving_env,
                'log_dir': os.path.join('logs', 'CrowdDriving',
                                       f'sac_discrete-{datetime.now().strftime("%Y%m%d-%H%M")}'),
                'gamma': 0.95,
                'multi_step': 1,
                'per': True,  # prioritized experience replay
                'alpha': 0.6,
                'beta': 0.4,
                'beta_annealing': 0.001,
                'cuda': True,
                'seed': args.seed,
                'batch_size': 64,
                'lr': args.lr,
                'memory_size': 1e5,
                'tau': 0.005,
                'target_update_interval': 1,
                'grad_clip': 5.0,
                'update_per_steps': 4,
                'log_interval': 100,
                'memory_load_interval': 5,
                'model_save_interval': 5,
                'model_checkpoint_interval': 30000,
                'eval_interval': 1000,
                'use_onpolicy_data': True,
                'entropy_annealing': 0.005,
                'entropy_anneal_interval': 3500, # 500,
                'policy_pt_delay': 0,
                'value_learning_delay': 0,
                'value_pt_delay': 0,  # 20000
                'start_steps': 10000,
                'termination_step': 300000
            }

            learner = SacDiscreteLearner(**config)
            learner.run()

    # env_server.send_signal(signal.SIGKILL)
    # env_server.wait(timeout=5)

    print_flush('[learner.py] termination')
