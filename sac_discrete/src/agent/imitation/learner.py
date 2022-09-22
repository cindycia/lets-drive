import os
from time import time
import numpy as np
import torch
from torch.optim import Adam
from datetime import datetime
from pathlib import Path
import Pyro4
import sys
import pickle

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent.parent
print(ws_root)
sys.stdout.flush()
SAVE_PATH = ws_root / 'sac_discrete' / 'trained_models'
sys.path.append(str(ws_root/'sac_discrete'/'src'))

from agent.imitation.base import ImitationAgent
from memory import DummyMultiStepMemory, DummyPrioritizedMemory
from policy import ConvCategoricalPolicy
from value import ConvVNetwork
from agent import soft_update, hard_update, update_params
from env import CrowdDriving
from env.planner_reward import reward as reward_func
from utils import data_host, log_port, replay_port, print_flush, log_flush, error_handler_with_log
from env.reward import explained_variance_score


class ImitationLearner(ImitationAgent):

    def __init__(self, env, log_dir, learner_id=0, per=True,
                 alpha=0.6, beta=0.4, beta_annealing=0.001,
                 batch_size=64, lr=0.0003, memory_size=1e5, use_onpolicy_data=False, gamma=0.99,
                 tau=0.005, multi_step=3, grad_clip=5.0, update_per_steps=4,
                 start_steps=1000, value_learning_delay=1000, value_pt_delay=1000, policy_pt_delay = 0,
                 log_interval=1, memory_load_interval=5,
                 model_save_interval=5, model_checkpoint_interval=1000,
                 eval_interval=1000, target_update_interval=1,
                 entropy_annealing=0.00252, entropy_anneal_interval=1000,
                 cuda=True, seed=0, termination_step=350000, offline_training=False):
        try:
            self.log_flag = 'imitation/lr_{}_bs_{}_ms_{}_seed_{}'.format(lr, batch_size, memory_size, seed)
            self.log_txt = open("leaner_log_{}.txt".format(learner_id), "w")
            self.tau = tau
            self.lr = lr
            self.batch_size = batch_size
            self.start_steps = start_steps
            self.gamma_n = gamma ** multi_step
            self.grad_clip = grad_clip
            self.update_per_steps = update_per_steps
            self.value_learning_delay = value_learning_delay
            self.value_pt_delay = value_pt_delay
            self.policy_pt_delay = policy_pt_delay
            self.log_interval = log_interval
            self.memory_load_interval = memory_load_interval
            self.model_save_interval = model_save_interval
            self.model_checkpoint_interval = model_checkpoint_interval
            self.target_update_interval = target_update_interval
            self.eval_interval = eval_interval
            self.entropy_annealing = entropy_annealing
            self.entropy_anneal_interval = entropy_anneal_interval
            self.termination_step = termination_step
            self.use_onpolicy_data = use_onpolicy_data
            self.per = per
            self.learner_id = str(learner_id)
            self.offline_training = offline_training
            if self.offline_training:
                self.start_steps = self.termination_step - 1
                self.termination_step = self.termination_step * 2
                self.eval_interval = 50000
                memory_size = self.termination_step

            self.writer = None
            self.env = env
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            self.device = torch.device(
                "cuda" if cuda and torch.cuda.is_available() else "cpu")

            log_flush(self.log_txt, 'device {}'.format(self.device))

            torch.autograd.set_detect_anomaly(True)

            self.policy = ConvCategoricalPolicy(
                self.env.observation_space.shape[0],
                self.env.semantic_observation_space.n,
                self.env.action_space.n).to(self.device)

            self.value = ConvVNetwork(
                self.env.observation_space.shape[0],
                self.env.semantic_observation_space.n).to(self.device)
            self.value_target = ConvVNetwork(
                self.env.observation_space.shape[0],
                self.env.semantic_observation_space.n).to(self.device).eval()
            self.critic = None

            hard_update(self.value_target, self.value)

            self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
            self.policy_optim = Adam(self.policy.parameters(), lr=lr, weight_decay=0.01)
            self.value_optim = Adam(self.value.parameters(), lr=lr, weight_decay=0.0001)
            # self.value_scheduller = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.value_optim, T_0=4000,
            #                             eta_min=0.000001)
            self.value_scheduller = torch.optim.lr_scheduler.ReduceLROnPlateau(self.value_optim, min_lr=0.0001,
                                                                               factor=0.63, patience=10000)
            self.target_entropy = np.log(self.env.action_space.n) * 0.98  # max_prob 0.163
            # self.min_target_entropy = np.log(self.env.action_space.n) * 0.51  # max_prob 0.6
            self.min_target_entropy = np.log(self.env.action_space.n) * 0.65  # max_prob 0.5
            # self.min_target_entropy = np.log(self.env.action_space.n) * 0.35  # max_prob 0.75
            if self.offline_training:
                self.target_entropy = self.min_target_entropy

            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.steps = 0
            self.save_weights(leaner_id=learner_id)

            if per:
                self.memory = DummyPrioritizedMemory(
                    memory_size, self.env.observation_space.shape,
                    self.env.semantic_observation_space.n,
                    (1,), self.device, gamma, multi_step,
                    alpha=alpha, beta=beta, beta_annealing=beta_annealing)
            else:
                self.memory = DummyMultiStepMemory(
                    memory_size, self.env.observation_space.shape,
                    self.env.semantic_observation_space.n,
                    (1,), self.device, gamma, multi_step)

            self.model_dir = str(SAVE_PATH / self.learner_id)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

            Pyro4.config.COMMTIMEOUT = 0.0  # infinite wait
            Pyro4.config.SERIALIZER = 'pickle'
            self.logging_service = Pyro4.Proxy('PYRO:logservice.warehouse@{}:{}'.format(data_host, log_port))

            Pyro4.config.SERIALIZER = 'pickle'
            log_flush(self.log_txt, '[learner.py] ' + 'Connecting to replay service at '
                                          'PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))
            self.replay_service = Pyro4.Proxy('PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))
            # self.replay_service._pyroAsync()
            self.last_replay_block = 0 

            self.epochs = 0

        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def __del__(self):
        self.log_txt.close()

    def run(self):
        try:
            self.load_memory()

            self.time = time()
            while len(self.memory) < self.start_steps:
                self.evaluate()
                self.load_memory()

            # self.time = time()
            while self.total_data_count() <= self.termination_step:
                self.epochs += 1
                for _ in range(self.update_per_steps):
                    self.steps += 1
                    self.learn()
                    self.interval()
        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def learn(self):
        try:
            log_flush(self.log_txt, 'step {}'.format(self.steps))
            if self.per:
                batch, indices, weights = \
                    self.memory.sample(self.batch_size)
            else:
                batch = self.memory.sample(self.batch_size)
                weights = 1.

            policy_loss, entropies, pi = self.calc_policy_loss(batch, weights)
            combined_loss, value_loss, col_loss, v_mask_loss, col_mask_loss, errors, \
            mean_v, mean_col, mean_target_v, mean_target_col, evar_v, evar_col, \
                mask_accuracy, mean_v_mask, mean_col_mask, mean_target_v_mask, mean_target_col_mask = \
                self.calc_value_loss(batch, weights)
            entropy_loss = self.calc_entropy_loss(entropies, weights)

            update_params(
                self.policy_optim, self.policy, policy_loss, self.grad_clip, retain_graph=True)
            if self.steps >= self.value_learning_delay:
                update_params(
                    self.value_optim, self.value, combined_loss, self.grad_clip, retain_graph=True)
                if self.steps == self.value_pt_delay:
                    self.value_scheduller._reset()
                    for g in self.value_optim.param_groups:
                        g['lr'] = self.lr
                # block whenusing constant lr
                # self.value_scheduller.step(value_loss + col_loss + v_mask_loss + col_mask_loss)
            # value_lr = self.value_scheduller.get_last_lr()[0]
            value_lr = self.value_optim.param_groups[0]['lr']

            # print_flush('value_lr={}'.format(value_lr))
            self.update_params(
                self.alpha_optim, None, entropy_loss)

            if self.per:
                self.memory.update_priority(indices, errors.cpu().numpy())

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
                    'loss/policy': policy_loss.detach().item(),
                    'loss/mask_accuracy': mask_accuracy,
                    'loss/evar_value_ncol': evar_v,
                    'loss/evar_value_col': evar_col,
                    # 'loss/alpha': entropy_loss.detach().item(),
                    # 'stats/alpha': self.alpha.detach().item(),
                    'stats/value_lr': value_lr,
                    'stats/mean_value_ncol': mean_v,
                    'stats/mean_value_ncol_target': mean_target_v,
                    'stats/mean_value_col': mean_col,
                    'stats/mean_value_col_target': mean_target_col,
                    'stats/mean_mask_ncol': mean_v_mask,
                    'stats/mean_mask_ncol_target': mean_target_v_mask,
                    'stats/mean_mask_col': mean_col_mask,
                    'stats/mean_mask_col_target': mean_target_col_mask,
                    'stats/entropy': entropies.detach().mean().item(),
                    'stats/target_entropy': self.target_entropy,
                    'stats/max_prob': pi_max.cpu().mean(),
                    'stats/policy_kl': policy_kl.mean(),
                    'stats/policy_grad_norm': policy_grad_norm,
                    'stats/value_grad_norm': value_grad_norm,
                    'progress/total_data_count': self.total_data_count()
                }
                self.logging_service.add_log(self.log_interval, self.log_flag, log_dict)
        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def calc_policy_loss(self, batch, weights):
        try:
            states, semantic_states, actions, _, _, _, _, _, _, _ = batch
            _, action_probs, log_action_probs, _ =\
                self.policy.sample(states, semantic_states)
            entropies = -torch.sum(
                action_probs * log_action_probs, dim=1, keepdim=True)
            policy_loss = self.ce_loss(action_probs, actions.squeeze(dim=1).long()) - self.alpha * entropies
            policy_loss = (policy_loss * weights).mean()

            return policy_loss, entropies, action_probs
        except Exception as e:
            error_handler_with_log(self.log_txt, e)

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

            # l = 5
            # print_flush('cur_col_mask={}\ntarget_col_mask={}\ncur_hard_mask={}\nmismatch={}'.format(
            #     cur_col_mask[:l], target_col_mask[:l], (cur_col_mask >= 0.5).float()[:l],
            #     ((cur_col_mask >= 0.5).float() != target_col_mask).float()[:l]))

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
            error_handler_with_log(self.log_txt, e)

    def calc_entropy_loss(self, entropies, weights):
        entropy_loss = -(
            self.log_alpha
            * (self.target_entropy - entropies).detach() * weights
            ).mean()
        return entropy_loss

    def interval(self):
        if self.steps % self.eval_interval == 0:
            self.evaluate()
            if not self.offline_training and self.steps > self.value_learning_delay:
                self.eval_interval = min(int(self.memory.size() * 0.2), 10000)
                print_flush('[learner] adjust eval_interval to {}'.format(self.eval_interval))
        if self.steps % self.memory_load_interval == 0:
            self.load_memory()
        if self.steps % self.model_save_interval == 0:
            self.save_weights(self.learner_id)
        if self.steps % self.model_checkpoint_interval == 0:
            self.save_models()
        if self.steps % self.target_update_interval == 0:
            soft_update(self.value_target, self.value, self.tau)
        if self.steps % self.entropy_anneal_interval == 0:
            if self.value_pt_delay < 1000000:
                if self.steps > self.value_pt_delay: # anneal entropy target only when using value for search
                    self.target_entropy = max(self.target_entropy - self.entropy_annealing, self.min_target_entropy)
            else:
                if self.total_data_count() >= self.start_steps: # anneal entropy target only when using value for search
                    self.target_entropy = max(self.target_entropy - self.entropy_annealing, self.min_target_entropy)

    def evaluate(self):
        try:
            # if self.offline_training and self.total_data_count() > self.start_steps:
            #     return

            self.save_pt()

            episodes = 5
            returns = np.zeros((episodes,), dtype=np.float32)

            step_vels = []
            collisions = []

            effective_episodes = 0
            while effective_episodes < episodes:
                log_flush(self.log_txt, 'eval episode {}'.format(effective_episodes))

                if 'imitation' in args.drive_mode:
                    state, semantic_state = self.env.reset(True, os.path.exists(str(SAVE_PATH / '0' / 'policy')))
                else:
                    state, semantic_state = self.env.reset(False, os.path.exists(str(SAVE_PATH / '0' / 'policy')))

                if state is None:
                    log_flush(self.log_txt, 'Environment reset failed. Wasting episode')
                    continue

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

                    if reward is None:
                        log_flush(self.log_txt, "current step output {} {} {} {} {} {}".format(
                            next_state, next_semantic_state, reward, collision, terminal, offpolicy_action
                        ))
                        raise Exception("reward is None")

                    episode_reward += reward
                    if variables is not None:
                        step_vels.append(variables['vel'])
                        collisions.append(variables['col'])
                        factored_value = [variables['value_ncol'], variables['value_col']]
                    else:
                        factored_value = None
                    if self.use_onpolicy_data:
                        self.update_memory(state, next_state, offpolicy_action, semantic_state, next_semantic_state,
                                           factored_value, terminal, reward, collision, episode_done, true_actor=True)
                    episode_length += 1
                    state = next_state
                    semantic_state = next_semantic_state

                if episode_length > 1:
                    returns[effective_episodes] = episode_reward
                    effective_episodes += 1

            mean_return = np.mean(returns)

            if self.total_data_count() > self.start_steps:
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
            error_handler_with_log(self.log_txt, e)

    def save_models(self):
        torch.save(self.policy.state_dict(), str(SAVE_PATH / self.learner_id / 'policy_cp_{}'.format(self.steps)))
        torch.save(self.value.state_dict(), str(SAVE_PATH / self.learner_id / 'value_cp_{}'.format(self.steps)))
        # self.value_target.save(
        #     os.path.join(self.model_dir, 'value_target.pth'))

    def save_weights(self, leaner_id):
        try:
            log_flush(self.log_txt, '[learner.py] save weights to {}'.format(str(SAVE_PATH / str(leaner_id))))
            leaner_id = str(leaner_id)
            if not os.path.isdir(str(SAVE_PATH / leaner_id)):
                os.makedirs(str(SAVE_PATH / leaner_id))

            if self.steps >= self.policy_pt_delay:
                torch.save(self.policy.state_dict(), str(SAVE_PATH / leaner_id / 'policy'))
                torch.save(self.value.state_dict(), str(SAVE_PATH / leaner_id / 'value'))
                torch.save(self.value_target.state_dict(), str(SAVE_PATH / leaner_id / 'value_target'))
                if self.steps >= self.value_learning_delay + self.value_pt_delay:
                    torch.save(torch.ones(1), str(SAVE_PATH / leaner_id / 'use_value'))

                pickle.dump(self.alpha.clone().detach().item(), (SAVE_PATH / leaner_id / 'alpha').open('wb'))
        except Exception as e:
            error_handler_with_log(self.log_txt, e)

    def save_pt(self):
        try:
            example_forward_input = (torch.rand(2, *self.env.observation_space.shape).to(self.device),
                                     torch.rand(2, self.env.semantic_observation_space.n).to(self.device))

            if os.path.exists(str(SAVE_PATH / self.learner_id / 'policy')):
                filename = str(ws_root / 'crowd_pomdp_planner' / 'temp_policy_net_{}.pt'.format(self.learner_id))
                log_flush(self.log_txt, '[learner.py] export ts policy to {} for evaluation'.format(filename))
                ts_policy = torch.jit.trace(self.policy, example_forward_input)
                torch.jit.save(ts_policy, filename)

            if os.path.exists(str(SAVE_PATH / self.learner_id / 'use_value')):
                filename = str(ws_root / 'crowd_pomdp_planner' / 'temp_value_net_{}.pt'.format(self.learner_id))
                if os.path.exists(filename):
                    log_flush(self.log_txt, '[learner.py] export ts value to {} for evaluation'.format(filename))
                    ts_value = torch.jit.trace(self.value, example_forward_input)
                    torch.jit.save(ts_value, filename)
            return True
        except Exception as e:
            log_flush(self.log_txt, e)
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
    parser.add_argument('--offline',
                        type=bool,
                        default=False,
                        help='offline training mode')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    use_data = False
    # if 'joint_pomdp' in args.drive_mode:
    #     use_data = True

    import subprocess
    with subprocess.Popen('python3 {} --port {}'.format(
        str(ws_root / 'sac_discrete' / 'src' / 'env' / 'env_service.py'), args.port).split()) as env_server:
        with CrowdDriving(summit_port=args.port, gpu_id=args.gpu, launch_env=args.env_mode,
                                drive_mode=args.drive_mode, record_bag=False, reward_func=reward_func) as driving_env:
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
                'memory_size': 1e5,
                'tau': 0.005,
                'grad_clip': 5.0,
                'update_per_steps': 4,
                'value_learning_delay': 0,
                'value_pt_delay': 0, # 20000
                'log_interval': 10,
                'memory_load_interval': 5,
                'model_save_interval': 100,
                'model_checkpoint_interval': 30000,
                'eval_interval': 1000,
                'use_onpolicy_data': use_data,
                'entropy_annealing': 0.005,
                'entropy_anneal_interval': 3500, # 500, 
                'target_update_interval': 1,
                'policy_pt_delay': 0,
                'start_steps': 10000,
                'termination_step': 300000,
                'offline_training': args.offline
            }

            learner = ImitationLearner(**config)
            learner.run()

    # env_server.send_signal(signal.SIGKILL)
    # env_server.wait(timeout=5)

    print_flush('[learner.py] termination')
