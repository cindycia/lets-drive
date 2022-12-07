import os
import rospy
import sys
import time
import numpy as np
from collections import OrderedDict

import eventlet
import eventlet.wsgi
import socketio
from flask import Flask
from msg_builder.srv import TensorData

import torch
import torch.nn as nn
from torch.autograd import Variable

model_device_id = 0
device = torch.device("cuda:" + str(model_device_id) if torch.cuda.is_available() else "cpu")

cwd = os.getcwd()
catkin_ws_src = os.path.dirname(os.path.dirname(cwd))
print('catkin_ws={}'.format(catkin_ws_src))
nn_folder = os.path.join(catkin_ws_src, 'il_controller/src/')
if not os.path.isdir(nn_folder):
    raise Exception('nn folder does not exist: {}'.format(nn_folder))

sys.path.append(os.path.join(nn_folder, 'Config'))
sys.path.append(nn_folder)

import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from Config.global_params import config
from transforms import VelEncoderRaw2Onehot
from Config.setup_params import parse_cmd_args, update_global_config, print_settings, load_settings_from_model
from Utils.error import error_handler

vel_encoder = VelEncoderRaw2Onehot()
sm = nn.Softmax(dim=1)


def clear_png_files(root_folder, subfolder=None, remove_flag='.png'):
    if subfolder:
        folder = root_folder + subfolder + '/'
    else:
        folder = root_folder

    if not os.path.exists(folder):
        os.makedirs(folder)

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path) and remove_flag in the_file:
                os.remove(file_path)
            if os.path.isdir(file_path):
                clear_png_files(root_folder=folder, subfolder=the_file, remove_flag=remove_flag)
        except Exception as e:
            error_handler(e)

step = 0
def on_data_ready(data):
    with torch.no_grad():
        print("Received data bs={}, depth={}".format(data.batchsize, data.mode), flush=True)
        bs = data.batchsize
        mode = data.mode
        # print("data.mode={}".format(mode), flush=True)
        cur_vel = np.asarray(data.cur_vel)
        # print("data.cur_vel={}".format(cur_vel), flush=True)

        start = time.time()
        # print("reshape data of length {}".format(len(data.tensor)), flush=True)
        data_size = [bs, config.total_num_channels, config.imsize, config.imsize]
        pt_tensor_from_list = torch.as_tensor(data.tensor).view(data_size).unsqueeze(1).view(data_size)
        input = pt_tensor_from_list.to(device)

        # print("reshape semantic data of length {}".format(len(data.semantic_tensor)), flush=True)
        data_size = [bs, config.num_hist_channels]
        pt_tensor_from_list = torch.as_tensor(data.semantic_tensor).view(data_size)
        semantic_input = pt_tensor_from_list.to(device)

        _, acc_logits, \
        ang_logits, vel_logits, lane_logits, _, _ = net.forward(input, semantic_input)
        global sm
        vel = sm(vel_logits)
        ang = sm(ang_logits)
        lane = sm(lane_logits)
        acc = cal_acc(cur_vel, lane, vel)

        # print(" lane={}\n acc={}".format(lane, acc), flush=True)

        ang = ang.cpu().view(-1).detach().numpy().tolist()
        vel = vel.cpu().view(-1).detach().numpy().tolist()
        lane = lane.cpu().view(-1).detach().numpy().tolist()
        acc = acc.cpu().view(-1).detach().numpy().tolist()

        value = np.zeros(int(len(lane)/3), dtype=float)

        depth = int(mode)
        if False: #depth == 5:
            print("Drawing input tensors", flush=True)
            img = input.cpu().detach().numpy()

            global step
            # plt.imsave("{}_agents.png".format(step), img[0][0])
            # plt.imsave("{}_lane.png".format(step), img[0][4])
            # img = np.swapaxes(img[0][2:5], 0, 2)
            print("agents image max={}".format(np.max(img[0, 0, :, :])), flush=True)
            print("lane image max={}".format(np.max(img[0, 4, :, :])), flush=True)

            img = img[0, (0,3,4)].transpose(1, 2, 0).astype(np.uint8)

            plt.imsave("{}_lane.png".format(step), img)
            step += 1

        print("at depth {}, model forward time: {}s".format(mode, str(time.time() - start)), flush=True)

    return value, acc, ang, vel, lane


def cal_acc(cur_vel, lane, vel):
    try:
        # print("Calculate acc", flush=True)
        global vel_encoder
        acc = torch.zeros(lane.shape)
        # print("shapes: acc {}, lane {}, vel {}, cur_vel {}".format(
        #     acc.shape, lane.shape, vel.shape, cur_vel.shape), flush=True)

        for i in range(vel.shape[0]):
            # print("i={}".format(i), flush=True)
            vel_i = vel[i]
            vel_idx = vel_encoder(cur_vel[i])
            # print("vel_idx = {}".format(vel_idx), flush=True)
            acc[i][0] = vel_i[0:vel_idx].sum()
            acc[i][1] = vel_i[vel_idx]
            acc[i][2] = vel_i[vel_idx + 1:].sum()
    except Exception as e:
        print(e, flush=True)
        error_handler(e)
        raise("Calculate acc error")
    return acc


def print_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("No. parameters in model: %d", params)


if __name__ == '__main__':
    print("main entry")

    clear_png_files(root_folder=os.getcwd())

    import policy_value_network

    model_mode = ''

    net = None

    if model_mode is '':
        cmd_args = parse_cmd_args()
        update_global_config(cmd_args)
        print("configuration done")
        print("=> loading checkpoint '{}'".format(cmd_args.modelfile))
        checkpoint = torch.load(os.path.join(nn_folder, cmd_args.modelfile))
        load_settings_from_model(checkpoint)
        print_settings(cmd_args)

        net = policy_value_network.PolicyValueNet()
        print_model_size(net)
        net = nn.DataParallel(net, device_ids=[0]).to(device)  # device_ids= config.GPU_devices
        net.load_state_dict(checkpoint['state_dict'])
        print("=> model at epoch {}".format(checkpoint['epoch']))

        X = torch.randn([1, 1, config.total_num_channels, config.imsize, config.imsize])
        X1 = torch.randn([1, config.num_hist_channels])

        # X = Variable(torch.randn([1, 1, config.total_num_channels, config.imsize, config.imsize]))
        value, acc, ang, vel, lane, _, _ = net.forward(X, X1)

    sys.stdout.flush()

    rospy.init_node('nn_query_node')
    s = rospy.Service('query', TensorData, on_data_ready)
    print("Ready to query nn.", flush=True)
    rospy.spin()
