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
