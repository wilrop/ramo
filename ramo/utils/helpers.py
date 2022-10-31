import warnings

import numpy as np
import scipy.optimize as scopt


def array_slice(array, axis, start, end, step=1):
    """Slice an array across a desired axis.

    Note:
        See: https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis

    Args:
        array (ndarray): An input array.
        axis (int): The axis to slice through.
        start (int): The start index of that axis.
        end (int): The end index of that axis.
        step (int, optional): The step size of the slice. (Default value = 1)

    Returns:
        ndarray: A slice across the correct axis.
    """
    return array[(slice(None),) * (axis % array.ndim) + (slice(start, end, step),)]


def in_hull(x, points):
    """Check whether a point is a convex combination of a set of points.

    Args:
        x (ndarray): The point to check.
        points (ndarray): An array of points.

    Returns:
        bool: Whether the point was in the convex hull.
    """
    n_points = len(points)  # The number of points.
    c = np.zeros(n_points)  # Make an array of zeros of this size as the objective to minimise.
    A = np.r_[points.T, np.ones((1, n_points))]  # Add row of ones such that the strategy sums to 1.
    b = np.r_[x, np.ones(1)]  # The strategy array.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=scopt.OptimizeWarning)  # Suppress full row rank warnings.
        lp = scopt.linprog(c, A_eq=A, b_eq=b)  # Check if we can find a convex combination by linear programming.
    return lp.success
