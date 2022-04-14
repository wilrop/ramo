import numpy as np


def random_monfg(player_actions=(2, 2), num_objectives=2, reward_min_bound=0, reward_max_bound=5, rng=None):
    """Generate a random MONFG with payoffs from a discrete uniform distribution.

    Args:
        player_actions (Tuple[int], optional): A tuple of actions indexed by player. (Default value = (2, 2))
        num_objectives (int, optional): The number of objectives in the game. (Default value = 2)
        reward_min_bound (int, optional): The minimum reward on an objective. (Default value = 0)
        reward_max_bound (int, optional): The maximum reward on an objective. (Default value = 5)
        rng (Generator, optional): A random number generator. (Default value = None)

    Returns:
        List[ndarray]: A list of payoff matrices representing the MONFG.

    """
    rng = rng if rng is not None else np.random.default_rng()
    payoffs = []
    payoffs_shape = player_actions + tuple([num_objectives])  # Define the shape of the payoff matrices.

    for _ in range(len(player_actions)):
        payoff_matrix = rng.integers(low=reward_min_bound, high=reward_max_bound, size=payoffs_shape)
        payoffs.append(payoff_matrix)

    return payoffs


def identity_game(player_actions):
    """Generate an identity game.

    Args:
        player_actions (Tuple[int]): A tuple of actions indexed by player.

    Returns:
        List[ndarray]: A list of payoff matrices representing the identity game.

    """
    payoffs = []
    joint_strat_length = np.sum(player_actions)  # Description length of a joint strategy.
    payoffs_shape = player_actions + tuple([joint_strat_length])  # Shape of the payoff matrices.
    payoff_matrix = np.zeros(payoffs_shape)  # Make the same payoff matrix for every player.

    for joint_strat in np.ndindex(player_actions):  # Loop over joint strategies.
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


def scalarised_game(monfg, u_tpl):
    """Scalarise an MONFG, which is a list of payoff matrices, using individual utility functions.

    Note:
        The scalarised game is sometimes referred to as a trade-off game.

    Args:
        monfg (List[ndarray]): A list of payoff matrices.
        u_tpl (Tuple[callable]): A utility function per player.

    Returns:
        List[ndarray]: The scalarised game. Each payoff matrix has the same shape as in the MONFG except the last
        dimension which is removed due to the scalarisation.

    """
    player_actions = monfg[0].shape[:-1]
    scalarised = []

    for payoff_matrix, u in zip(monfg, u_tpl):
        scalarised_payoffs = np.zeros(player_actions)

        for idx in np.ndindex(player_actions):  # Loop over all possible strategies.
            utility = u(payoff_matrix[idx])
            scalarised_payoffs[idx] = utility

        scalarised.append(scalarised_payoffs)
    return scalarised
