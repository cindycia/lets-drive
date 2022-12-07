import os
import signal
import subprocess
import time
import sys
import gym
import numpy as np
from gym import spaces
from pathlib import Path
import cv2

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from env.reward import reward as default_reward_func
from utils import print_flush, data_host, env_port, error_handler
import Pyro4


def launch_carla(port, gpu, mode='server'):
    CARLA_PATH = os.path.expanduser("~/summit/CarlaUE4.sh")
    print("CARLA_PATH: ",CARLA_PATH)
    print_flush("[crowd_driving_env.py] Launching carla at port {}, usng gpu {}".format(port, gpu))

    try:
        if mode == 'server':
            #print_flush("[crowd_driving_env.py] carla in server mode")
            proc = subprocess.Popen(
                [
                    CARLA_PATH,
                    '-carla-port={}'.format(port),
                    '-quality-level=Low',
                    '-RenderOffScreen'
                ],
                env=dict(os.environ, 
                #SDL_VIDEODRIVER='offscreen', 
                SDL_HINT_CUDA_DEVICE='{}'.format(gpu)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid)
            #print_flush("[crowd_driving_env.py] carla in server mode")
        else:
            print_flush("[crowd_driving_env.py] carla in display mode")
            proc = subprocess.Popen(
                [
                    CARLA_PATH,
                    '-opengl',
                    '-carla-port={}'.format(port),
                    '-quality-level=Low'
                ],
                env=dict(os.environ, DISPLAY='', SDL_HINT_CUDA_DEVICE='{}'.format(gpu)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid)
        print_flush("[crowd_driving_env.py] carla launched")
        return proc
    except Exception as e:
        print_flush("[crowd_driving_env.py] exception when launching summit: {}".format(e), flush=True)
        exit("summit launch error")


class CrowdDriving(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, summit_port, gpu_id, reward_func=default_reward_func,
                 launch_env='server', record_bag=False, drive_mode='lets-drive-zero'):
        super(CrowdDriving, self).__init__()

        self.port = summit_port
        self.gpu = gpu_id
        self.launch_mode = launch_env
        if self.launch_mode == 'display':
            self.gpu = 0
        else:
            print_flush('launch_mode is {}_end'.format(self.launch_mode))
        self.record = record_bag
        self.drive_mode = drive_mode
        self.carla_proc = None
        self.docker_proc = None
        self.docker_out = None

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(9)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(5, 64, 64), dtype=np.uint8)
        self.semantic_observation_space = spaces.Discrete(4)

        self.reward_func = reward_func
        self.cur_state = None
        self.last_data_update = None
        self.next_variables = None

        Pyro4.config.SERIALIZER = 'pickle'
        print_flush('[crowd_driving_env] Connecting to env service...')
        self.env_service = Pyro4.Proxy('PYRO:envservice.warehouse@{}:{}'.format(data_host, env_port + summit_port))
        self.env_service._pyroAsync()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #print(exc_type, exc_value, traceback)
        self.close()

    def step(self, on_pol_action=None):
        try:
            # Execute one time step within the environment
            initialized, key, data_point = self.env_service.pop_data_point().value

            start = time.time()
            while initialized is True and key is None:
                poll = self.docker_proc.poll()
                if poll is None and time.time() - start < 30.0:  # 30.0
                    time.sleep(0.01)
                    initialized, key, data_point = self.env_service.pop_data_point().value
                else:
                    self.end_episode()
                    return None, None, 0.0, None, None, None, None

            if initialized is False:
                while initialized is False:
                    if time.time() - start < 50.0:
                        time.sleep(0.01)
                        initialized, key, data_point = self.env_service.pop_data_point().value
                    else:
                        print_flush("waited for too long to start an episode, ending it instead.")
                        self.end_episode()
                        return None, None, 0.0, None, None, None, None
                print_flush('[crowd_driving_env.py] initial step')
            else:
                pass # print_flush('[crowd_driving_env.py] step')

            self.last_data_update = time.time()
            obs = data_point.next_state
            obs_semantic = data_point.next_state_semantic
            off_pol_action = data_point.action
            variables = self.next_variables
            self.next_variables = data_point.next_variables

            if self.launch_mode == 'display' and 'lets-drive-' in self.drive_mode:
                image = obs.astype(np.uint8)
                new_image_red, new_image_green, new_image_blue = image[-2,...], image[0,...], image[-1,...]
                self.cur_state = np.dstack([new_image_red, new_image_green, new_image_blue])
                self.cur_state = cv2.resize(self.cur_state, (0,0), fx=4.0, fy=4.0)

            if off_pol_action is not None:
                if variables is None or obs is None:
                    print_flush("[crowd_driving_env.py] corrupted data (None) get from step function, ending episode.")
                    self.end_episode()
                    return None, None, 0.0, None, None, None, None
                reward = self.reward_func(off_pol_action, variables['vel'], variables['ttc'],
                                                     variables['is_terminal'], variables['col'])
                return obs, obs_semantic, reward, variables['col'], variables['is_terminal'], off_pol_action, variables
            else:
                return obs, obs_semantic, 0.0, None, None, None, None
        except Exception as e:
            error_handler(e)

    def end_episode(self):
        print("[crowd_driving_env] end_episode.")
        self.next_variables = None
        if self.launch_mode == 'display':
            cv2.destroyAllWindows()
        # subprocess.call('pkill -P ' + str(carla_proc.pid), shell=True)
        self.kill_simulator()
        self.env_service.reset()

    def kill_simulator(self):
        print_flush('[crowd_driving_env.py] kill docker')
        if self.last_data_update is not None and time.time() - self.last_data_update > 200.0:
            print_flush('[crowd_driving_env.py] env have collapsed, '
                        'clearing the oldest CarlaUE4-Linux- processes')
            subprocess.call("timeout 5 pkill -9 -o CarlaUE4-Linux-", shell=True)
        try:
            if self.docker_proc:
                self.docker_proc.kill()
                outs, errs = self.docker_proc.communicate(timeout=15)
                print_flush('[crowd_driving_env.py] kill docker outs {}, errs {}'.format(outs, errs))
                poll = self.docker_proc.poll()
                while poll is None:
                    os.kill(self.docker_proc.pid, signal.SIGKILL)
                    time.sleep(1)
                self.docker_proc = None
                if self.docker_out:
                    self.docker_out.close()
        except Exception as e:
            print_flush('[crowd_driving_env.py] kill docker ' + str(e))

        if self.carla_proc is not None:
            print_flush('[crowd_driving_env.py] kill carla')
            try:
                while self.carla_proc.poll() is None:
                    os.killpg(os.getpgid(self.carla_proc.pid), signal.SIGKILL)
                    time.sleep(1)
            except Exception as e:
                error_handler('[crowd_driving_env.py] kill carla ' + str(e))

    def reset(self, evaluation=False, prior_ready=False):
        self.env_service.reset()
        self.start_episode(evaluation, prior_ready)

        # Reset the state of the environment to an initial state
        obs, obs_semantic, _, _, _, _, _ = self.step()  # the step will produce a None action
        return obs, obs_semantic

    def start_episode(self, evaluation, prior_ready):
        if prior_ready and 'labeller' in self.drive_mode:
            self.carla_proc = None
        else:
            self.carla_proc = launch_carla(self.port, self.gpu, self.launch_mode)
        print_flush('[crowd_driving_env.py] Waiting for CARLA to start...')
        #print(self.carla_proc)
        time.sleep(10)
        if evaluation is False:
            drive_mode = self.drive_mode
        else:
            drive_mode = 'imitation'
        self.launch_planner(drive_mode)
        print_flush('[crowd_driving_env.py] new episode started')

    def launch_planner(self, drive_mode):
        if self.launch_mode == 'server':
            shell_cmd = 'python launch_docker.py --port {} --gpu {} --recordbag {} --mode {}'.format(
                self.port, self.gpu, int(self.record), drive_mode)
            print_flush("[crowd_driving_env.py] Ececuting: " + shell_cmd)
            self.docker_proc = subprocess.Popen(shell_cmd.split())
            # self.docker_out = open('docker_log.txt', 'w')
            # self.docker_proc = subprocess.Popen(shell_cmd.split(), stdout=self.docker_out, stderr=self.docker_out)
        else:
            shell_cmd = 'bash experiment_summit.sh {} {} 0 {} 0 0 {}'.format(
                drive_mode, self.gpu, int(self.record), self.port)
            print_flush("[crowd_driving_env.py] Ececuting: " + shell_cmd)
            self.docker_proc = subprocess.Popen(shell_cmd.split())

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if self.launch_mode == 'display' and 'lets-drive-' in self.drive_mode:
            if self.cur_state is not None:
                cv2.imshow("cur_state", self.cur_state)
                cv2.waitKey(1)

    def close(self):
        print_flush('[crowd_driving_env.py] close')
        if self.launch_mode == 'display':
            cv2.destroyAllWindows()
        self.kill_simulator()


carla_proc, docker_proc = None, None

import atexit
@atexit.register
def goodbye():
    try:
        if docker_proc:
            docker_proc.kill()
            outs, errs = docker_proc.communicate(timeout=15)
            print_flush('[crowd_driving_env.py] kill docker outs {}, errs {}'.format(outs, errs))
            poll = docker_proc.poll()
            while poll is None:
                os.kill(docker_proc.pid, signal.SIGKILL)
                time.sleep(1)
    except Exception as e:
        print_flush('[crowd_driving_env.py] kill docker ' + str(e))
    if carla_proc is not None:
        print_flush('[crowd_driving_env.py] kill carla')
        try:
            while carla_proc.poll() is None:
                os.killpg(os.getpgid(carla_proc.pid), signal.SIGKILL)
                time.sleep(1)
        except Exception as e:
            error_handler('[crowd_driving_env.py] kill carla ' + str(e))


if __name__ == '__main__':
    drive_mode = sys.argv[1]
    carla_proc = launch_carla(2000, 0, 'display')
    shell_cmd = 'bash experiment_summit.sh {} {} 0 {} 0 0 {}'.format(
        drive_mode, 0, 0, 2000)
    print_flush("[crowd_driving_env.py] Ececuting: " + shell_cmd)
    docker_proc = subprocess.Popen(shell_cmd.split(), cwd=str(ws_root / 'scripts'))
    time.sleep(10000)
