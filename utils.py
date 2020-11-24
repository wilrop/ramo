import os
import errno
import numpy as np


def mkdir_p(path):
    """
    This function makes a new directory at the given path.
    :param path: The path to the new directory.
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def create_game_path(content, criterion, game, opt_init, rand_prob):
    """
    This function will create a new directory based on the given parameters.
    :param content: The type of content this directory will hold.
    :param criterion: The multi-objective optimisation criterion.
    :param game: The current game that is being played.
    :param opt_init: A boolean that decides on optimistic initialization of the Q-tables.
    :param rand_prob: A boolean that decides on random initialization for the mixed strategy.
    :return: The path that was created.
    """
    path = f'{content}/{criterion}/{game}'

    if opt_init:
        path += '/opt_init'
    else:
        path += '/zero_init'

    if rand_prob:
        path += '/opt_rand'
    else:
        path += '/opt_eq'

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
