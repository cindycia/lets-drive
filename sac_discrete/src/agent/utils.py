import torch
import sys


def to_batch(state, state_semantic, action, reward, collision, next_state, next_state_semantic, factored_value, done, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    state_semantic = torch.FloatTensor(state_semantic).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    next_state_semantic = torch.FloatTensor(next_state_semantic).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    collision = torch.FloatTensor([collision]).unsqueeze(0).to(device)
    ncol_value = torch.FloatTensor([factored_value[0]]).unsqueeze(0).to(device)
    col_value = torch.FloatTensor([factored_value[1]]).unsqueeze(0).to(device)

    return state, state_semantic, action, reward, collision, next_state, next_state_semantic, ncol_value, col_value, done


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
