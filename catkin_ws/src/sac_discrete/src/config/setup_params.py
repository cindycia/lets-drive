import argparse

import numpy

from config.global_params import config


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str,
                        default=None, help='Path to train file')
    parser.add_argument('--val', type=str,
                        default=None, help='Path to val file')
    parser.add_argument('--imsize',
                        type=int,
                        default=config.imsize,
                        help='Size of input image')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,
                        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--epochs',
                        type=int,
                        default=config.num_epochs,
                        help='Number of epochs to train')
    parser.add_argument('--moreepochs',
                        type=int,
                        default=-1,
                        help='Number of more epochs to train (only used when in resume mode)')
    parser.add_argument('--start_epoch',
                        type=int,
                        default=0,
                        help='start epoch for training')
    parser.add_argument('--k',
                        type=int,
                        default=config.num_gppn_iterations,  # 10
                        help='Number of Value Iterations')
    parser.add_argument('--f',
                        type=int,
                        default=config.gppn_kernelsize,
                        help='Number of Value Iterations')
    parser.add_argument('--l_i',
                        type=int,
                        default=config.num_gppn_inputs,
                        help='Number of channels in input layer')
    parser.add_argument('--l_h',
                        type=int,
                        default=config.num_gppn_hidden_channels,  # 150
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q',
                        type=int,
                        default=10,
                        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size',
                        type=int,
                        default=config.batch_size,
                        help='Batch size')
    parser.add_argument('--l2',
                        type=float,
                        default=config.l2_reg_weight,
                        help='Weight decay')
    parser.add_argument('--no_action',
                        type=int,
                        default=4,
                        help='Number of actions')
    parser.add_argument('--modelfile',
                        type=str,
                        default="drive_net_params",
                        help='Name of model file to be saved')
    parser.add_argument('--exactname',
                        type=int,
                        default=0,
                        help='Use exact model file name as given')
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Name of model file to be loaded for resuming training')
    parser.add_argument('--no_vin',
                        type=int,
                        default=int(config.vanilla_resnet),
                        help='Number of pedistrians')
    parser.add_argument('--fit',
                        type=str,
                        default='action',
                        help='Number of pedistrians')
    parser.add_argument('--w',
                        type=int,
                        default=config.resnet_width,
                        help='ResNet Width')
    parser.add_argument('--vinout',
                        type=int,
                        default=config.gppn_out_channels,
                        help='ResNet Width')
    parser.add_argument('--nres',
                        type=int,
                        default=config.num_resnet_layers,
                        help='Number of resnet layers')
    parser.add_argument('--goalfile',
                        type=str,
                        default="../../Maps/indian_cross_goals_15_10_15.txt",
                        help='Name of model file to be saved')
    parser.add_argument('--ssm',
                        type=float,
                        default=config.sigma_smoothing,
                        help='smoothing scalar added on to sigma predictions in mdn heads')
    parser.add_argument('--trainpath',
                        type=str,
                        default='',
                        help='path for the set of training h5 files')
    parser.add_argument('--logdir',
                        type=str,
                        default='runs/**CURRENT_DATETIME_HOSTNAME**',
                        help='path for putting tensor board logs')
    parser.add_argument('--goalx',
                        type=float,
                        default=config.car_goal[0],
                        help='goal_x for car')
    parser.add_argument('--goaly',
                        type=float,
                        default=config.car_goal[1],
                        help='goal_y for car')
    parser.add_argument('--v_scale',
                        type=float,
                        default=config.val_scale,
                        help='scale up of value loss')
    parser.add_argument('--ang_scale',
                        type=float,
                        default=config.ang_scale,
                        help='scale up of value loss')
    parser.add_argument('--do_p',
                        type=float,
                        default=config.do_prob,
                        help='drop out prob')
    parser.add_argument('--input_model',
                        type=str,
                        default='',
                        help='[model conversion] Input model in pth format')
    parser.add_argument('--output_model',
                        type=str,
                        default="torchscript_version.pt",
                        help='[model conversion] Output model in pt format')
    parser.add_argument('--monitor',
                        type=str,
                        default="data_monitor",
                        help='which data monitor to use: data_monitor or summit_dql')
    parser.add_argument('--net',
                        type=str,
                        default='not specified',
                        help='which network to train')

    return parser.parse_args()


def update_global_config(cmd_args):
    print("\n=========================== Command line arguments ==========================")
    for arg in vars(cmd_args):
        print("==> {}: {}".format(arg, getattr(cmd_args, arg)))
    print("=========================== Command line arguments ==========================\n")

    # Update the global configurations according to command line
    config.batch_size = cmd_args.batch_size
    config.l2_reg_weight = cmd_args.l2
    config.vanilla_resnet = bool(cmd_args.no_vin)
    config.num_gppn_inputs = cmd_args.l_i
    config.num_gppn_hidden_channels = cmd_args.l_h
    config.gppn_kernelsize = cmd_args.f
    config.num_gppn_iterations = cmd_args.k
    config.resnet_width = cmd_args.w
    config.gppn_out_channels = cmd_args.vinout
    config.sigma_smoothing = cmd_args.ssm
    print("Sigma smoothing " + str(cmd_args.ssm))
    config.train_set_path = cmd_args.trainpath
    config.num_resnet_layers = cmd_args.nres

    if not config.train_set_path == '':
        config.sample_mode = 'hierarchical'
    else:
        config.sample_mode = 'random'

    print("Using sampling mode " + str(config.sample_mode))

    config.car_goal[0] = float(cmd_args.goalx)
    config.car_goal[1] = float(cmd_args.goaly)
    config.val_scale = cmd_args.v_scale
    config.ang_scale = cmd_args.ang_scale
    config.do_prob = cmd_args.do_p

    if "not specified" in cmd_args.net:
        set_fit_mode_bools(cmd_args)
    elif "policy" in cmd_args.net:
        specify_policy_net_setting()
    elif "value" in cmd_args.net:
        specify_value_net_setting()


def print_settings(cmd_args):
    print("Fitting " + cmd_args.fit)
    print_global_config()


def print_global_config():
    print("\n=========================== Global configuration ==========================")
    for arg in vars(config):
        print("===> {}: {}".format(arg, getattr(config, arg)))
    print("Channels: agent channels {}-{}, lane channel {}".format(
        config.channel_map[0], config.channel_map[-1], config.channel_lane))
    print("=========================== Global configuration ==========================\n")


def set_fit_mode_bools(cmd_args):
    if cmd_args.fit == 'action':
        config.fit_action = True
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = False
        config.fit_qvalue = False
    elif cmd_args.fit == 'acc':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = True
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = False
        config.fit_qvalue = False
    elif cmd_args.fit == 'steer':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = True
        config.fit_val = False
        config.fit_lane = False
        config.fit_qvalue = False
    elif cmd_args.fit == 'vel':
        config.fit_action = False
        config.fit_vel = True
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = False
        config.fit_qvalue = False
        config.use_vel_head = True
    elif cmd_args.fit == 'lane':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = True
        config.fit_qvalue = False
    elif cmd_args.fit == 'val':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = True
        config.fit_lane = False
        config.fit_qvalue = False
    elif cmd_args.fit == 'all':
        config.fit_action = False
        config.fit_vel = False
        config.fit_acc = False
        config.fit_ang = False
        config.fit_val = False
        config.fit_lane = False
        config.fit_qvalue = True


def load_settings_from_model(checkpoint):
    try:
        config.vanilla_resnet = checkpoint['novin']
        config.num_gppn_hidden_channels = checkpoint['l_h']
        config.num_gppn_iterations = checkpoint['k']
        config.gppn_kernelsize = checkpoint['f']
        config.resnet_width = checkpoint['w']
        config.gppn_out_channels = checkpoint['vin_out']
        config.resblock_in_layers = checkpoint['layers']
        config.num_resnet_layers = checkpoint['n_res']
        config.head_mode = checkpoint['head_mode']
        config.do_prob = checkpoint['do_p']
        config.num_guassians_in_heads = checkpoint['n_modes']
        config.sigma_smoothing = checkpoint['ssm']
        config.disable_bn_in_resnet = checkpoint['disable_bn']
        config.track_running_stats = checkpoint['track_run_stat']
        config.fit_ang = checkpoint['fit_ang']
        config.fit_acc = checkpoint['fit_acc']
        config.fit_vel = checkpoint['fit_vel']
        config.fit_val = checkpoint['fit_val']
        config.fit_lane = checkpoint['fit_lane']
        config.fit_action = checkpoint['fit_action']
        config.fit_qvalue = checkpoint['fit_qvalue']
        if config.fit_vel:
            config.use_vel_head = True
    except Exception as e:
        print(e)
    finally:
        '''
        print("=> model at epoch {}"
              .format(checkpoint['epoch']))
        print("=> best val loss {}"
              .format(checkpoint['best_prec1']))
        print("=> head mode {}"
              .format(config.head_mode))
        print("=> vanilla resnet {}"
              .format(config.vanilla_resnet))
        print("=> width of vin {}"
              .format(config.num_gppn_hidden_channels))
        print("=> out channels of vin {}"
              .format(config.gppn_out_channels))
        print("=> resnet width {}"
              .format(config.resnet_width))
        print("=> res layers {}"
              .format(config.resblock_in_layers))
        print("=> mdn sigma smoothing {}"
              .format(config.sigma_smoothing))
        print("=> track running status {}"
              .format(config.track_running_stats))
        '''
        pass


def specify_policy_net_setting():
    config.fit_action = True
    config.fit_val = False
    config.fit_vel = False
    config.fit_acc = False
    config.fit_ang = False
    config.fit_lane = False
    config.fit_qvalue = False
    config.use_vel_head = False
    config.num_vel_bins = 8
    config.resnet_width = 32
    config.vanilla_resnet = True
    config.do_dropout = False
    config.num_resnet_layers = 2
    config.use_leaky_relu = True


def specify_value_net_setting():
    config.fit_action = False
    config.fit_val = True
    config.fit_vel = False
    config.fit_acc = False
    config.fit_ang = False
    config.fit_lane = False
    config.fit_qvalue = False
    config.use_vel_head = False
    config.resnet_width = 32
    config.vanilla_resnet = True
    config.do_dropout = False
    config.num_resnet_layers = 2
    config.use_leaky_relu = True


def specify_qvalue_net_setting():
    config.fit_action = False
    config.fit_val = False
    config.fit_vel = False
    config.fit_acc = False
    config.fit_ang = False
    config.fit_lane = False
    config.fit_qvalue = True
    config.use_vel_head = False
    config.resnet_width = 32
    config.vanilla_resnet = True
    config.do_dropout = False
    config.num_resnet_layers = 2
    config.use_leaky_relu = True
