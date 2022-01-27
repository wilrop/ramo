def u1(vector):
    """This function calculates the utility by doing x^2 + y^2. This is a convex function.

    Args:
      vector (ndarray): A payoff vector.

    Returns:
      float: The scalar utility for this vector.

    """
    utility = vector[0] ** 2 + vector[1] ** 2
    return utility


def u2(vector):
    """This function calculates the utility by doing x*y. This prefers the most balanced payoff vector.

    Args:
      vector (ndarray): A payoff vector.

    Returns:
      float: The scalar utility for this vector.

    """
    utility = vector[0] * vector[1]
    return utility


def u3(vector):
    """This function calculates the utility by doing x*y - y^2.

    Args:
      vector (ndarray): A payoff vector.

    Returns:
      float: The scalar utility for this vector.

    """
    utility = vector[0] * vector[1] - vector[1] ** 2
    return utility


def u4(vector):
    """A utility function that is a constant.

    Args:
      vector (ndarray): A payoff vector.

    Returns:
      float: A constant utility k.

    """
    k = 2
    return k


def u5(vector):
    """This function calculates the utility by doing x^2 + x*y + y^2. This is a convex function.

    Args:
      vector (ndarray): A payoff vector.

    Returns:
      float: The scalar utility for this vector.

    """
    utility = vector[0] ** 2 + vector[0] * vector[1] + vector[1] ** 2
    return utility


def u6(vector):
    """This function calculates the utility by doing x^2 + y. This is a convex function.

    Args:
      vector (ndarray): A payoff vector.

    Returns:
      float: The scalar utility for this vector.

    """
    utility = vector[0] ** 2 + vector[1]
    return utility


def u7(vector):
    """This function calculates the utility by doing x + y^2. This is a convex function.

    Args:
      vector (ndarray): A payoff vector.

    Returns:
      float: The scalar utility for this vector.

    """
    utility = vector[0] + vector[1] ** 2
    return utility


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
    else:
        raise Exception(f'The provided utility function "{u_str}" does not exist.')
