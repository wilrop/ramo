import numpy as np


def pareto_dominates(a, b):
    """Check if the vector in a Pareto dominates vector b.

    Args:
        a (ndarray): A numpy array.
        b (ndarray): A numpy array.

    Returns:
        bool: Whether vector a dominates vector b.
    """
    a = np.array(a)
    b = np.array(b)
    return np.all(a >= b) and np.any(a > b)


def strict_pareto_dominates(a, b):
    """Check if the vector in a Pareto strictly dominates vector b.

    Args:
        a (ndarray): A numpy array.
        b (ndarray): A numpy array.

    Returns:
        bool: Whether vector a strictly dominates vector b.
    """
    a = np.array(a)
    b = np.array(b)
    return np.all(a > b)
