import os
from os.path import expanduser
import subprocess

home = expanduser("~")

catkin_ws_path = home + '/lets_drive_docker/catkin_ws'

summit_path = home + "/summit"

if not os.path.isdir(catkin_ws_path):
    catkin_ws_path = home + '/catkin_ws'

result_path = home + '/driving_data'


# check whether the folders exist:
if not os.path.isdir(result_path):
    os.makedirs(result_path)
    print("Made result folder " + result_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--image',
                        type=str,
                        default="wenjingtang/noetic_cuda11_1_cudnn8_libtorch_opencv4_ws",
                        help='Image to launch')
    parser.add_argument('--port',
                        type=int,
                        default=2000,
                        help='carla_port')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='GPU to use')
    parser.add_argument('--recordbag',
                        type=int,
                        default=0,
                        help='record ros bags')
    parser.add_argument('--mode',
                        type=str,
                        default='lets-drive-zero',
                        help='driving mode') 

    config = parser.parse_args()

    additional_mounts = "-v " + catkin_ws_path + ":/root/catkin_ws -v " + summit_path + ":/root/summit "

    cmd_args = "docker run --rm --runtime=nvidia -it --network host " + \
                "-v " + result_path + ":/root/driving_data " + \
                                additional_mounts + \
                "-e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix " + \
                config.image + " " + str(config.gpu) + " " + str(config.port) \
                + " " + str(config.recordbag) + " " + str(config.mode)
    print(cmd_args)
    subprocess.call(cmd_args.split())
