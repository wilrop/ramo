import numpy as np

# Game 1: The (im)balancing act game. There are no NE under SER.
monfg1 = [np.array([[(4, 0), (3, 1), (2, 2)],
                    [(3, 1), (2, 2), (1, 3)],
                    [(2, 2), (1, 3), (0, 4)]], dtype=float),
          np.array([[(4, 0), (3, 1), (2, 2)],
                    [(3, 1), (2, 2), (1, 3)],
                    [(2, 2), (1, 3), (0, 4)]], dtype=float)]

# Game 2: The (im)balancing act game without M. There are no NE under SER.
monfg2 = [np.array([[(4, 0), (2, 2)],
                    [(2, 2), (0, 4)]], dtype=float),
          np.array([[(4, 0), (2, 2)],
                    [(2, 2), (0, 4)]], dtype=float)]

# Game 3: The (im)balancing act game without R. (L, M) is a pure NE under SER.
monfg3 = [np.array([[(4, 0), (3, 1)],
                    [(3, 1), (2, 2)]], dtype=float),
          np.array([[(4, 0), (3, 1)],
                    [(3, 1), (2, 2)]], dtype=float)]

# Game 4: A two action game. There are NE under SER for (L,L) and (M,M).
monfg4 = [np.array([[(4, 1), (1, 2)],
                    [(3, 1), (3, 2)]], dtype=float),
          np.array([[(4, 1), (1, 2)],
                    [(3, 1), (3, 2)]], dtype=float)]

# Game 5: A three action game. There are  NE under SER for (L,L), (M,M) and (R,R).
monfg5 = [np.array([[(4, 1), (1, 2), (2, 1)],
                    [(3, 1), (3, 2), (1, 2)],
                    [(1, 2), (2, 1), (1, 3)]], dtype=float),
          np.array([[(4, 1), (1, 2), (2, 1)],
                    [(3, 1), (3, 2), (1, 2)],
                    [(1, 2), (2, 1), (1, 3)]], dtype=float)]

# Game 6: A multi-objectivised version of the game of chicken. We use the utility function u2 for both players.
# The cyclic equilibrium is to go 2/3 your own, 1/3 other action uniformly over these.
monfg6 = [np.array([[(0, 0), (7, 2)],
                    [(2, 7), (6, 2.32502)]], dtype=float),
          np.array([[(0, 0), (2, 7)],
                    [(7, 2), (6, 2.32502)]], dtype=float)]

# Game 7: An example of a game where commitment may be exploited.
monfg7 = [np.array([[(-1, -1), (-1, 1)],
                    [(1, -1), (1, 1)]], dtype=float),
          np.array([[(-1, -1), (-1, 1)],
                    [(1, -1), (1, 1)]], dtype=float)]

# Game 8: A two action game. There are two NE when both agents use utility function u2 under SER: (L,L) and  (R, R).
# The cyclic equilibrium is to mix uniformly over these.
monfg8 = [np.array([[(10, 2), (0, 0)],
                    [(0, 0), (2, 10)]], dtype=float),
          np.array([[(10, 2), (0, 0)],
                    [(0, 0), (2, 10)]], dtype=float)]

# Game 9: A noisy version of game 8.
monfg9 = [np.array([[(10, 2), (2, 3)],
                    [(4, 2), (6, 3)]], dtype=float),
          np.array([[(10, 2), (2, 3)],
                    [(4, 2), (6, 3)]], dtype=float)]

# Game 10: A game without Nash equilibria that still has a cyclic Nash equilibrium.
monfg10 = [np.array([[(2, 0), (1, 0)],
                    [(0, 1), (0, 2)]], dtype=float),
           np.array([[(2, 0), (1, 1)],
                    [(1, 1), (0, 2)]], dtype=float)]

# Game 11: The same game as game 10 but intended to be used with the utility functions reversed.
monfg11 = [np.array([[(2, 0), (1, 0)],
                    [(0, 1), (0, 2)]], dtype=float),
           np.array([[(2, 0), (1, 1)],
                    [(1, 1), (0, 2)]], dtype=float)]


def u1(vector):
    """
    This function calculates the utility for agent 1.
    :param vector: The reward vector.
    :return: The utility for agent 1.
    """
    utility = vector[0] ** 2 + vector[1] ** 2
    return utility


def gradient_u1(vector):
    """
    This function returns the partial derivative for the two objectives for agent 1.
    :param vector: The reward vector.
    :return: An array of the two partial derivatives for agent 1.
    """
    dx = 2 * vector[0]
    dy = 2 * vector[1]
    return np.array([dx, dy])


def u2(vector):
    """
    This function calculates the utility for agent 2.
    :param vector: The reward vector.
    :return: The utility for agent 2.
    """
    utility = vector[0] * vector[1]
    return utility


def gradient_u2(vector):
    """
    This function returns the partial derivative for the two objectives for agent 2.
    :param vector: The reward vector.
    :return: An array of the two partial derivatives for agent 2.
    """
    dx = vector[1]
    dy = vector[0]
    return np.array([dx, dy])


def u3(vector):
    """
    This function calculates the utility from a vector.
    :param vector: The reward vector.
    :return: The utility from this vector.
    """
    utility = vector[0] * vector[1] - vector[1] ** 2  # i.e. balanced
    return utility


def gradient_u3(vector):
    """
    This function returns the partial derivative for the two objectives for utility function 3.
    :param vector: The reward vector.
    :return: An array of the two partial derivatives.
    """
    dx = vector[1]  # = y
    dy = vector[0] - 2 * vector[1]  # = x - 2y
    return np.array([dx, dy])


def u4(vector):
    """
    A utility function that is a constant.
    :param vector: The input payoff vector.
    :return: A constant utility k.
    """
    k = 2
    return k


def gradient_u4(vector):
    """
    This function returns the partial derivative for the two objectives for utility function 4.
    :param vector: The reward vector.
    :return: An array of the two partial derivatives.
    """
    dx = 0
    dy = 0
    return np.array([dx, dy])


def get_monfg(game):
    """
    This function will provide the correct payoffs based on the game we play.
    :param game: The current game.
    :return: A list of payoff matrices.
    """
    if game == 'game1':
        monfg = monfg1
    elif game == 'game2':
        monfg = monfg2
    elif game == 'game3':
        monfg = monfg3
    elif game == 'game4':
        monfg = monfg4
    elif game == 'game5':
        monfg = monfg5
    elif game == 'game6':
        monfg = monfg6
    elif game == 'game7':
        monfg = monfg7
    elif game == 'game8':
        monfg = monfg8
    elif game == 'game9':
        monfg = monfg9
    elif game == 'game10':
        monfg = monfg10
    elif game == 'game11':
        monfg = monfg11
    else:
        raise Exception("The provided game does not exist.")

    return monfg


def get_u_and_du(u_str):
    """
    Get the utility function and derivative of the utility function.
    :param u_str: The string of this utility function.
    :return: A utility function and derivative.
    """
    if u_str == 'u1':
        return u1, gradient_u1
    elif u_str == 'u2':
        return u2, gradient_u2
    elif u_str == 'u3':
        return u3, gradient_u3
    elif u_str == 'u4':
        return u4, gradient_u4
    else:
        raise Exception('The provided utility function does not exist.')


def scalarise_matrix(payoff_matrix, u):
    """
    This function scalarises an input matrix using a provided utility function.
    :param payoff_matrix: An input payoff matrix.
    :param u: A utility function.
    :return: The trade-off game.
    """
    scalarised_matrix = np.zeros((payoff_matrix.shape[0], payoff_matrix.shape[1]))
    for i in range(scalarised_matrix.shape[0]):
        for j in range(scalarised_matrix.shape[1]):
            utility = u(payoff_matrix[i, j])
            scalarised_matrix[i, j] = utility
    return scalarised_matrix
