import Pyro4
import random
import numpy as np
import cv2
from pathlib import Path
import os
import time
import sys

ws_root = Path(os.path.realpath(__file__)).parent.parent.parent.parent
# sys.path.append(str(ws_root / 'il_controller' / 'src'))
sys.path.append(str(ws_root / 'sac_discrete' / 'src'))

from utils import data_host, replay_port, error_handler
import numpy
import matplotlib.pyplot as plt


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    # ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    binwidth = xymax/50.0
    lim = (int(xymax/binwidth) + 1) * binwidth

    #bins = np.arange(-lim, lim + binwidth, binwidth)
    bins = np.arange(-0.1, lim + binwidth, binwidth)
    # ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pos',
                        type=int,
                        default=0,
                        help='pos of data point')
    config = parser.parse_args()

    Pyro4.config.SERIALIZER = 'pickle'
    replay_service = Pyro4.Proxy('PYRO:replayservice.warehouse@{}:{}'.format(data_host, replay_port))

    # try:
    #     last_replay_block = config.pos
    #     y_data = []
    #     x_range = []
    #     for i in range(50):
    #         last_replay_block, new_memory = replay_service.fetch_memory(last_replay_block)
    #         # print('factored_value={}'.format(new_memory['factored_value']))
    #         col_list = new_memory['factored_value'][..., 1]
    #         y_data = y_data + [abs(ele) for ele in col_list]
    #         x_range = x_range + [i/30.0] * len(col_list)
    #
    #     print('x {} \n y {}'.format(x_range, y_data))
    #     # fig = plt.figure()
    #     # ax = fig.add_axes([0, 0, 1, 1])
    #     # ax.set_xlabel('x')
    #     # ax.set_ylabel('y')
    #     # ax.set_title('scatter plot')
    #     # ax.scatter(x_range, y_data, color='r')
    #
    #     left, width = 0.1, 0.65
    #     bottom, height = 0.1, 0.65
    #     spacing = 0.005
    #
    #     rect_scatter = [left, bottom, width, height]
    #     # rect_histx = [left, bottom + height + spacing, width, 0.2]
    #     rect_histy = [left + width + spacing, bottom, 0.2, height]
    #
    #     # start with a square Figure
    #     fig = plt.figure(figsize=(8, 8))
    #
    #     ax = fig.add_axes(rect_scatter)
    #     # ax_histx = fig.add_axes(rect_histx, sharex=ax)
    #     ax_histy = fig.add_axes(rect_histy, sharey=ax)
    #     scatter_hist(x_range, y_data, ax, None, ax_histy)
    #
    #     plt.savefig('col_scatter.png')
    # except Exception as e:
    #     error_handler(e)

    last_replay_block, new_memory = replay_service.fetch_memory(config.pos)
    value_list = new_memory['factored_value']
    print('values: {}'.format(value_list))
    state_list = new_memory['state']
    for pos, state in enumerate(state_list):
        state_image = np.empty((5, 64, 64), dtype=np.uint8)
        state_image[...] = np.array(state, dtype=np.uint8)

        for i in range(5):
            # if i == 3:
            #     combined = np.empty((64, 64), dtype=np.uint8)
            #     cv2.max(state_image[4, ...], state_image[0, ...], combined)
            #     new_image_red, new_image_green, new_image_blue = combined, combined, combined
            # else:
            #     new_image_red, new_image_green, new_image_blue = state_image[i, ...], state_image[i+1, ...], state_image[i+1, ...]
            state_image[i, ...] = (255 - state_image[i, ...])
            new_image_red, new_image_green, new_image_blue = state_image[i, ...], state_image[i, ...], state_image[i, ...]
            cur_state = np.dstack([new_image_red, new_image_green, new_image_blue])
            cur_state = cv2.resize(cur_state, (0, 0), fx=4.0, fy=4.0)
            cv2.imshow("cur_state_{}".format(i), cur_state)
            if i == 4:
                cv2.imwrite("lane_{}.png".format(pos), cur_state)
            else:
                cv2.imwrite("cur_state_{}_{}.png".format(pos, i), cur_state)
            cv2.waitKey(1)
            time.sleep(0.33)
