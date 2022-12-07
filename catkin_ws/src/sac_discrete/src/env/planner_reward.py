import math
import numpy as np

from utils import error_handler
from env.reward import NUM_ACC, MAX_VEL, ACC_SPEED, CONTROL_FREQ, ACC_PROFILE

REWARD_FACTOR_VEL = 4.0


def get_lane(action):
    lane_ID = int(int(action) / NUM_ACC)
    return lane_ID - 1


def get_acc(action):
    acc_ID = action % NUM_ACC
    return ACC_PROFILE[str(int(acc_ID))]


def action_penalty(action):
    reward = 0.0
    acc = get_acc(action)
    if acc < 0.1:
        reward -= 0.1

    lane = get_lane(action)
    if lane != 0:
        reward -= REWARD_FACTOR_VEL
    return reward


def collision_penalty(vel):
    return - 3000.0 * (vel + 0.5)


def movement_penalty(vel):
    return min(REWARD_FACTOR_VEL * (vel - MAX_VEL) / MAX_VEL, 0.0)


def reward(action, vel, ttc, is_term, collision):
    try:
        reward = 0.0
        reward += action_penalty(action)
        reward += movement_penalty(vel)
        if collision:
            reward += collision_penalty(vel)
        return reward

    except Exception as e:
        error_handler(e)

