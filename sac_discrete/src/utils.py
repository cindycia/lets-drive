import torch
import sys
import traceback
from datetime import datetime


data_host = "127.0.0.1"
param_port = 7101
replay_port = 7102
log_port = 7103
env_port = 7104
labeller_port = 7105
virtual_replay_port = 7106

def error_handler(e):
    print(
        '\nError on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)
    print('Call-stack:')
    traceback.print_stack()
    sys.stdout.flush()
    # exit(-1)


def error_handler_with_log(file, e):
    error_handler(e)
    log_flush(file,
        '\nError on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename,
                                            sys.exc_info()[-1].tb_lineno))
    # exit(-1)


def print_flush(msg):
    print(msg)
    sys.stdout.flush()


def log_flush(file, msg):
    try:
        file.write('{}: {}\n'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), msg))
        file.flush()
        print_flush(msg)
    except Exception as e:
        error_handler(e)


def to_batch(state, action, reward, next_state, done, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    # torch.autograd.set_detect_anomaly(True)
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
        # for p in network.modules():
        #     try:
        #         torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
        #     except Exception as e:
        #         print(e)
        #         print("p: {}".format(p))
        #         sys.stdout.flush()
        #         raise e
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())

