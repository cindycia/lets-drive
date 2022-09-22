import atexit
import time
import os
import subprocess
import signal

carla_proc = None
docker_proc = None


def launch_carla(port, sport, gpu):
    CARLA_PATH = os.path.expanduser("~/summit/CarlaUE4.sh")
    print("Launching carla at port {}, usng gpu {}".format(port, gpu))

    try:
        # proc = subprocess.Popen(
        #     [
        #         CARLA_PATH, 
        #         '-carla-port={}'.format(port),
        #         '-quality-level=Low'
        #     ],
        #     env=dict(os.environ, SDL_VIDEODRIVER='offscreen', SDL_HINT_CUDA_DEVICE='{}'.format(gpu)),
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     preexec_fn=os.setsid)

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
        print("carla launched")
        return proc
    except Exception as e:
        print("exception when launching summit: {}".format(e), flush=True)
        exit("summit launch error") 


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='GPU to use')

    parser.add_argument('--trials',
                                            type=int,
                                            default=1,
                                            help='number of trials')

    parser.add_argument('--port',
                                            type=int,
                                            default=0,
                                            help='carla port')

    parser.add_argument('--mode',
                                            type=str,
                                            default='lets-drive-zero',
                                            help='driving mode')

    parser.add_argument('--record',
                                            type=int,
                                            default=0,
                                            help='record bags')

 
    config = parser.parse_args()

    if config.port == 0:
        config.port = 2000 + int(config.gpu)*1000
        config.sport = int(config.port) + 1

    carla_proc = None
    @atexit.register
    def goodbye():
        print("Exiting server pipeline.")
        if carla_proc:
            os.killpg(os.getpgid(carla_proc.pid), signal.SIGKILL)
            time.sleep(1)

    for trial in range(config.trials):

        carla_proc = launch_carla(config.port, config.sport, config.gpu)
        print('Waiting for CARLA to start...')
        time.sleep(4)
        
        shell_cmd = 'python launch_docker.py --port {} --gpu {} --recordbag {} --mode {}'.format(
                config.port, config.gpu, config.record, config.mode)
        print("Ececuting: "+shell_cmd)
        docker_proc = subprocess.call(shell_cmd, shell = True)

        print("Docker exited, closing simulator.")
        # subprocess.call('pkill -P ' + str(carla_proc.pid), shell=True)
        os.killpg(os.getpgid(carla_proc.pid), signal.SIGKILL)
        time.sleep(1)


