import numpy as np


def u1(vector):
    """Calculate the utility from: :math:`u(x, y) = x^2 + y^2`. This is a convex function.

    Args:
        vector (ndarray): A payoff vector.

    Returns:
        float: The scalar utility for this vector.

    """
    utility = vector[0] ** 2 + vector[1] ** 2
    return utility


def u2(vector):
    """Calculate the utility from: :math:`u(x, y) = x \cdot y`. This prefers the most balanced payoff vector.

    Args:
        vector (ndarray): A payoff vector.

    Returns:
        float: The scalar utility for this vector.

    """
    utility = vector[0] * vector[1]
    return utility


def u3(vector):
    """Calculate the utility from: :math:`u(x, y) = x \cdot y - y^2`.

    Args:
        vector (ndarray): A payoff vector.

    Returns:
        float: The scalar utility for this vector.

    """
    utility = vector[0] * vector[1] - vector[1] ** 2
    return utility


def u4(vector):
    """Calculate a constant utility. The constant is currently set to 2.

    Args:
        vector (ndarray): A payoff vector.

    Returns:
        float: A constant utility :math:`k`.

    """
    k = 2
    return k


def u5(vector):
    """Calculate the utility from: :math:`u(x, y) = x^2 + x \cdot y + y^2`. This is a convex function.

    Args:
        vector (ndarray): A payoff vector.

    Returns:
        float: The scalar utility for this vector.

    """
    utility = vector[0] ** 2 + vector[0] * vector[1] + vector[1] ** 2
    return utility


def u6(vector):
    """Calculate the utility from: :math:`u(x, y) = x^2 + y`. This is a convex function.

    Args:
        vector (ndarray): A payoff vector.

    Returns:
        float: The scalar utility for this vector.

    """
    utility = vector[0] ** 2 + vector[1]
    return utility


def u7(vector):
    """Calculate the utility from: :math:`u(x, y) = x + y^2`. This is a convex function.

    Args:
        vector (ndarray): A payoff vector.

    Returns:
        float: The scalar utility for this vector.

    """
    utility = vector[0] + vector[1] ** 2
    return utility


def sum_u(vector):
    """Calculate the utility from: :math:`u(\overrightarrow{p}) = \sum p_i`.

    Args:
        vector (ndarray): A payoff vector.

    Returns:
        float: The scalar utility for this vector.

    """
    return np.sum(vector)


def product_u(vector):
    """Calculate the utility from: :math:`u(\overrightarrow{p}) = \prod p_i`.

    Args:
        vector (ndarray): A payoff vector.

    Returns:
        float: The scalar utility for this vector.

    """
    return np.prod(vector)


def get_u(u_str):
    """Get the utility function from a string.

    Args:
        u_str (str): The string of the utility function.

    Returns:
        callable: A utility function.

    """
    if u_str == 'u1':
        return u1
    elif u_str == 'u2':
        return u2
    elif u_str == 'u3':
        return u3
    elif u_str == 'u4':
        return u4
    elif u_str == 'u5':
        return u5
    elif u_str == 'u6':
        return u6
    elif u_str == 'u7':
        return u7
    elif u_str == 'sum_u':
        return sum_u
    elif u_str == 'product_u':
        return product_u
    else:
        raise Exception(f'The provided utility function "{u_str}" does not exist.')
