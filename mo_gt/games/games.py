import numpy as np

# Game 1: The (im)balancing act game. A 3-action 2-player game with team rewards.
# There are no NE under SER using u1 and u2.
# There are two PSNE using u1 and u5: [0, 0], [2, 2]. Checked for correctness using Gambit.
monfg1 = [np.array([[(4, 0), (3, 1), (2, 2)],
                    [(3, 1), (2, 2), (1, 3)],
                    [(2, 2), (1, 3), (0, 4)]], dtype=float),
          np.array([[(4, 0), (3, 1), (2, 2)],
                    [(3, 1), (2, 2), (1, 3)],
                    [(2, 2), (1, 3), (0, 4)]], dtype=float)]

# Game 2: The (im)balancing act game without M. A 2-action 2-player game with team rewards.
# There are no NE under SER using u1 and u2.
# There are two PSNE using u1 and u5: [0, 0], [1, 1]. Checked for correctness using Gambit.
monfg2 = [np.array([[(4, 0), (2, 2)],
                    [(2, 2), (0, 4)]], dtype=float),
          np.array([[(4, 0), (2, 2)],
                    [(2, 2), (0, 4)]], dtype=float)]

# Game 3: The (im)balancing act game without R. A 2-action 2-player game with team rewards.
# There is one NE under SER using u1 and u2: (L, M).
# There is one PSNE using u1 and u5: [0, 0]. Checked for correctness using Gambit.
monfg3 = [np.array([[(4, 0), (3, 1)],
                    [(3, 1), (2, 2)]], dtype=float),
          np.array([[(4, 0), (3, 1)],
                    [(3, 1), (2, 2)]], dtype=float)]

# Game 4: A 2-action 2-player game with team rewards.
# There are two NE under SER using u1 and u2: (L,L) and (M,M).
# There are two PSNE using u1 and u5: [0, 0], [1, 1]. Checked for correctness using Gambit.
# This game shows cyclic behaviour under IBR with simultaneous updates but not with alternating updates.
# This game shows no cyclic behaviour with fictitious play.
monfg4 = [np.array([[(4, 1), (1, 2)],
                    [(3, 1), (3, 2)]], dtype=float),
          np.array([[(4, 1), (1, 2)],
                    [(3, 1), (3, 2)]], dtype=float)]

# Game 5: A 3-action 2-player game with team rewards.
# There are three NE under SER using u1 and u2: (L,L), (M,M) and (R,R).
# There are three PSNE using u1 and u5: [0, 0], [1, 1], [2, 2]. Checked for correctness using Gambit.
# This game shows cyclic behaviour under IBR with simultaneous updates but not with alternating updates.
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

# Game 7: A 2-action 2-player game with team rewards.
# An example of a game where commitment may be exploited.
monfg7 = [np.array([[(-1, -1), (-1, 1)],
                    [(1, -1), (1, 1)]], dtype=float),
          np.array([[(-1, -1), (-1, 1)],
                    [(1, -1), (1, 1)]], dtype=float)]

# Game 8: A 2-action 2-player game with team rewards.
# There are two NE when both agents use utility function u2 under SER: (L,L) and (R, R).
# The cyclic equilibrium is to mix uniformly over these.
monfg8 = [np.array([[(10, 2), (0, 0)],
                    [(0, 0), (2, 10)]], dtype=float),
          np.array([[(10, 2), (0, 0)],
                    [(0, 0), (2, 10)]], dtype=float)]

# Game 9: A 2-action 2-player game with team rewards.
# A noisy version of game 8.
# The cyclic equilibrium with utility function u2 is to play A 75% of the time and 25% B.
monfg9 = [np.array([[(10, 2), (2, 3)],
                    [(4, 2), (6, 3)]], dtype=float),
          np.array([[(10, 2), (2, 3)],
                    [(4, 2), (6, 3)]], dtype=float)]

# Game 10: A 2-action 2-player game with individual rewards.
# This game has no Nash equilibrium with utility functions u1 and u2, but does have a cyclic Nash equilibrium.
monfg10 = [np.array([[(2, 0), (1, 0)],
                     [(0, 1), (0, 2)]], dtype=float),
           np.array([[(2, 0), (1, 1)],
                     [(1, 1), (0, 2)]], dtype=float)]

# Game 11: A 2-action 2-player game with individual rewards.
# The same game as game 10 but intended to be used with the utility functions reversed.
monfg11 = [np.array([[(2, 0), (1, 1)],
                     [(1, 1), (0, 2)]], dtype=float),
           np.array([[(2, 0), (1, 0)],
                     [(0, 1), (0, 2)]], dtype=float)]

# Game 12: A 3-action 2-player game with individual rewards.
# This game has two PSNE using u1 and u5: [0, 0], [2, 2]. Checked for correctness using Gambit.
monfg12 = [
    np.array([[(4, 1), (1, 2), (2, 1)],
              [(3, 1), (3, 2), (1, 2)],
              [(1, 2), (2, 1), (1, 3)]], dtype=float),
    np.array([[(4, 0), (3, 1), (2, 2)],
              [(3, 1), (2, 2), (1, 3)],
              [(2, 2), (1, 3), (0, 4)]], dtype=float)
]

# Game 13: A 3-action 2-player game with individual rewards.
# This game has no PSNE using u1 and u5. Checked for correctness using Gambit.
monfg13 = [
    np.array([[(2, 3), (3, 2), (1, 1)],
              [(2, 5), (0, 2), (5, 2)],
              [(1, 3), (4, 0), (1, 3)]], dtype=float),
    np.array([[(0, 3), (1, 2), (2, 1)],
              [(2, 2), (3, 2), (1, 2)],
              [(3, 1), (0, 3), (1, 0)]], dtype=float)
]

# Game 14: A 2-action 3-player game with individual rewards.
# This game has two PSNE using u1, u5 and u6: [0, 1, 1], [1, 0, 1]. Checked for correctness by hand.
monfg14 = [
    np.array([[[(1, 0), (2, 1)],
               [(3, 0), (1, 2)]],
              [[(0, 2), (2, 2)],
               [(3, 1), (2, 0)]]], dtype=float),
    np.array([[[(2, 0), (0, 2)],
               [(1, 1), (1, 2)]],
              [[(0, 0), (1, 2)],
               [(2, 1), (0, 0)]]], dtype=float),
    np.array([[[(1, 2), (2, 1)],
               [(0, 1), (2, 2)]],
              [[(1, 1), (0, 3)],
               [(1, 1), (1, 2)]]], dtype=float)
]

# Game 15: A 3-player game where player 1 has 3 actions, player 2 has 2 and player 3 has 3, with individual rewards.
# This game has three PSNE using u1, u5 and u6: [0, 1, 1], [1, 0, 2], [1, 1, 0]. Checked for correctness by hand.
monfg15 = [
    np.array([[[(1, 0), (2, 1), (1, 2)],
               [(3, 0), (1, 2), (2, 2)]],
              [[(0, 2), (2, 2), (3, 0)],
               [(3, 1), (2, 0), (0, 1)]],
              [[(1, 1), (0, 0), (2, 1)],
               [(1, 2), (2, 0), (3, 0)]]]),
    np.array([[[(0, 2), (0, 1), (1, 1)],
               [(1, 3), (2, 2), (2, 2)]],
              [[(0, 2), (2, 0), (3, 0)],
               [(3, 1), (1, 0), (2, 1)]],
              [[(2, 2), (2, 1), (2, 0)],
               [(0, 1), (1, 3), (1, 1)]]]),
    np.array([[[(1, 3), (1, 1), (2, 2)],
               [(2, 1), (2, 3), (2, 0)]],
              [[(0, 2), (1, 1), (3, 1)],
               [(3, 1), (2, 1), (2, 1)]],
              [[(0, 1), (1, 0), (0, 0)],
               [(1, 1), (2, 1), (1, 1)]]])
]


def get_monfg(game):
    """Get the payoffs for a game from a string.

    Args:
      game (str): The string of the game.

    Returns:
      List[ndarray]: A list of payoff matrices.

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
    elif game == 'game12':
        monfg = monfg12
    elif game == 'game13':
        monfg = monfg13
    elif game == 'game14':
        monfg = monfg14
    elif game == 'game15':
        monfg = monfg15
    else:
        raise Exception("The provided game does not exist.")

    return monfg


def scalarise_matrix(payoff_matrix, u):
    """Scalarise a payoff matrix using a given utility function.

    Args:
      payoff_matrix (ndarray): The input payoffs.
      u (callable): A utility function.

    Returns:
      (ndarray): The scalarised game. This has the same shape except the last dimension which is scalarised.

    """
    player_actions = payoff_matrix.shape[:-1]
    scalarised_matrix = np.zeros(player_actions)
    num_strategies = np.prod(player_actions)

    for i in range(num_strategies):  # Loop over all possible strategies.
        idx = np.unravel_index(i, player_actions)  # Get the strategy from the flat index.
        utility = u(payoff_matrix[idx])
        scalarised_matrix[idx] = utility

    return scalarised_matrix


def generate_random_monfg(player_actions=(2, 2), num_objectives=2, reward_min_bound=0, reward_max_bound=5):
    """Generate a random MONFG with payoffs from a discrete uniform distribution.

    Args:
      player_actions (Tuple[int], optional): A tuple of actions indexed by player. (Default value = (2, 2))
      num_objectives (int, optional): The number of objectives in the game. (Default value = 2)
      reward_min_bound (int, optional): The minimum reward on an objective. (Default value = 0)
      reward_max_bound (int, optional): The maximum reward on an objective. (Default value = 5)

    Returns:
      List[ndarray]: A list of payoff matrices representing the MONFG.

    """
    payoffs = []
    payoffs_shape = player_actions + tuple([num_objectives])  # Define the shape of the payoff matrices.

    for _ in range(len(player_actions)):
        payoff_matrix = np.random.randint(low=reward_min_bound, high=reward_max_bound, size=payoffs_shape)
        payoffs.append(payoff_matrix)

    return payoffs


def generate_identity_game(player_actions=(2, 2)):
    """Generate an identity game.

    Args:
      player_actions (Tuple[int]): A tuple of actions indexed by player. (Default value = (2, 2))

    Returns:
      List[ndarray]: A list of payoff matrices representing the identity game.

    """
    payoffs = []
    joint_strat_length = np.sum(player_actions)  # Description length of a joint strategy.
    num_joint_strat = np.prod(player_actions)  # Number of joint strategies.
    payoffs_shape = player_actions + tuple([joint_strat_length])  # Shape of the payoff matrices.
    payoff_matrix = np.zeros(payoffs_shape)  # Make the same payoff matrix for every player.

    for flat_joint_strat in range(num_joint_strat):  # Loop over joint strategies.
        joint_strat = np.unravel_index(flat_joint_strat, player_actions)  # Get the coordinates.
        identity_vec = []  # Initialise the identity payoff. One hot encode joint strategies in this variable.
        for player, action in enumerate(joint_strat):  # One hot encode each player's strategy.
            strat_vec = np.zeros(player_actions[player])
            strat_vec[action] = 1
            identity_vec.extend(list(strat_vec))
        payoff_matrix[joint_strat] = np.array(identity_vec)

    payoffs.append(payoff_matrix)

    for _ in range(len(player_actions) - 1):  # We already have the first payoff matrix, so copy the rest now.
        payoff_copy = np.copy(payoff_matrix)
        payoffs.append(payoff_copy)

    return payoffs
