import os
import numpy as np


def mkdir_p(path):
    """
    This function makes a new directory at the given path.
    :param path: The path to the new directory.
    :return:
    """
    os.makedirs(path, exist_ok=True)


def create_game_path(content, experiment, game, opt_init):
    """
    This function will create a new directory based on the given parameters.
    :param content: The type of content this directory will hold.
    :param experiment: The name of the experiment that is being performed.
    :param game: The current game that is being played.
    :param opt_init: A boolean that decides on optimistic initialization of the Q-tables.
    :return: The path that was created.
    """
    path = f'{content}/{experiment}/{game}'

    if opt_init:
        path += '/opt_init'
    else:
        path += '/zero_init'

    return path


def softmax(q):
    """
    Calculates the softmax function over an input array.
    :param q: The input array.
    :return: The softmax function of the input.
    """
    soft_q = np.exp(q - np.max(q))
    return soft_q / soft_q.sum(axis=0)


def softmax_grad(softmax):
    """
    This function calculates the Jacobian of the softmax input vector.
    J[i][j] = s[i] * (1 - s[j]) if i == j
    J[i][j] = -s[i] * s[j] if i != j
    The implementation below is a very short implementation to give the resulting Jacobian.
    :param softmax: The input softmax vector.
    :return: The derivative of this function. This is the actually the Jacobian.
    """
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)
