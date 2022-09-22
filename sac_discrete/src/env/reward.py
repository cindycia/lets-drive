import math
import numpy as np

from utils import error_handler

NUM_ACC = 3
MAX_VEL = 6.0
ACC_SPEED = 3.0
CONTROL_FREQ = 3.0

LANE_REWARD = -0.025
VEL_REWARD_SCALE = 0.05
TTC_REWARD_SCALE = 1.0
TERM_REWARD = -1.0

ACC_PROFILE = {
    '0': 0.0,
    '1': 1.0,
    '2': -1.0
}


def ttc_reward(ttc):
    if ttc > MAX_VEL / ACC_SPEED + 1.0: # 3 seconds
        return 0.0
    else:
        return - math.pow(1.0 / (ttc * CONTROL_FREQ), 2)


def get_lane(action):
    lane_ID = int(int(action) / NUM_ACC)
    return lane_ID - 1


def get_acc(action):
    acc_ID = action % NUM_ACC
    return ACC_PROFILE[str(int(acc_ID))]


def reward(action, vel, ttc, is_term, collision):
    try:
        reward = 0.0

        if is_term:
            # print('r_term {}, '.format(TERM_REWARD))
            return TERM_REWARD

        if collision:
            # print('r_col {}, '.format(TERM_REWARD))
            return TERM_REWARD

        lane = get_lane(action)
        if lane != 0:
            # print('r_lane {}, '.format(LANE_REWARD))
            reward += LANE_REWARD
        # print('r_vel {}, '.format(VEL_REWARD_SCALE * vel / MAX_VEL))
        reward += VEL_REWARD_SCALE * vel / MAX_VEL
        # print('r_tcc {}, '.format(TTC_REWARD_SCALE * ttc_reward(ttc)))
        reward += TTC_REWARD_SCALE * ttc_reward(ttc)
        return reward

    except Exception as e:
        error_handler(e)
        # sys.exit(0)


def check_consistent_length(*arrays):
    lengths = [X.shape[0] for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    check_consistent_length(y_true, y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    return y_true, y_pred, multioutput


def explained_variance_score(y_true, y_pred,
                             sample_weight=None,
                             multioutput='uniform_average'):
    """Explained variance regression score function
    Best possible score is 1.0, lower values are worse.
    Read more in the :ref:`User Guide <explained_variance_score>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average', \
                'variance_weighted'] or array-like of shape (n_outputs)
        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.
    Returns
    -------
    score : float or ndarray of floats
        The explained variance or ndarray if 'multioutput' is 'raw_values'.
    Notes
    -----
    """
    try:
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        y_true, y_pred, multioutput = _check_reg_targets(
            y_true, y_pred, multioutput)

        y_diff_avg = np.average(y_true - y_pred, weights=sample_weight, axis=0)
        numerator = np.average((y_true - y_pred - y_diff_avg) ** 2,
                               weights=sample_weight, axis=0)

        y_true_avg = np.average(y_true, weights=sample_weight, axis=0)
        denominator = np.average((y_true - y_true_avg) ** 2,
                                 weights=sample_weight, axis=0)

        nonzero_numerator = numerator != 0
        nonzero_denominator = denominator != 0
        valid_score = nonzero_numerator & nonzero_denominator

        output_scores = np.ones(y_true.shape[1])

        output_scores[valid_score] = 1 - (numerator[valid_score] /
                                          denominator[valid_score])
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
        if isinstance(multioutput, str):
            if multioutput == 'raw_values':
                # return scores individually
                return output_scores
            elif multioutput == 'uniform_average':
                # passing to np.average() None as weights results is uniform mean
                avg_weights = None
            elif multioutput == 'variance_weighted':
                avg_weights = denominator
        else:
            avg_weights = multioutput

        return np.average(output_scores, weights=avg_weights)
    except Exception as e:
        error_handler(e)

if __name__ == "__main__":
    import torch
    y_true = torch.rand(128)
    y_pred = y_true + 10.0 + torch.rand(128) * 0.1
    # y_pred = torch.rand(10)
    # y_true = [3, -0.5, 2, 7]
    # y_pred = [2.5, 0.0, 2, 8]
    print("y_true: {}\n y_pred: {}\n exp_var: {}".format(y_true, y_pred, explained_variance_score(y_true, y_pred)))
