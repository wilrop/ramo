import numpy as np

from ramo.game.properties import get_player_actions


def discrete_uniform_monfg(player_actions=(2, 2), num_objectives=2, reward_min_bound=0, reward_max_bound=5, rng=None):
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
    monfg = []
    payoffs_shape = player_actions + tuple([num_objectives])  # Define the shape of the payoff matrices.

    for _ in range(len(player_actions)):
        payoff_matrix = rng.integers(low=reward_min_bound, high=reward_max_bound, size=payoffs_shape)
        monfg.append(payoff_matrix)

    return monfg


def uniform_monfg(player_actions=(2, 2), num_objectives=2, reward_min_bound=0, reward_max_bound=5, rng=None):
    """Generate a random MONFG with payoffs from a uniform distribution.

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
    monfg = []
    payoffs_shape = player_actions + tuple([num_objectives])  # Define the shape of the payoff matrices.

    for _ in range(len(player_actions)):
        payoff_matrix = rng.uniform(low=reward_min_bound, high=reward_max_bound, size=payoffs_shape)
        monfg.append(payoff_matrix)

    return monfg


def normal_distributed_monfg(player_actions=(2, 2), num_objectives=2, mean=0, std=1, rng=None):
    """Generate a random MONFG with payoffs from a normal distribution.

    Args:
        player_actions (Tuple[int], optional): A tuple of actions indexed by player. (Default value = (2, 2))
        num_objectives (int, optional): The number of objectives in the game. (Default value = 2)
        mean (float, optional): The mean of the normal distribution. (Default value = 0)
        std (float, optional): The standard deviation of the normal distribution. (Default value = 1)
        rng (Generator, optional): A random number generator. (Default value = None)

    Returns:
        List[ndarray]: A list of payoff matrices representing the MONFG.

    """
    rng = rng if rng is not None else np.random.default_rng()
    monfg = []
    payoffs_shape = player_actions + tuple([num_objectives])  # Define the shape of the payoff matrices.

    for _ in range(len(player_actions)):
        payoff_matrix = rng.normal(loc=mean, scale=std, size=payoffs_shape)
        monfg.append(payoff_matrix)

    return monfg


def covariance_monfg(player_actions=(2, 2), num_objectives=2, mean=0, std=1, rho=0, rng=None):
    """Generate a random MONFG with payoffs from a normal distribution and given covariance.

    Args:
        player_actions (Tuple[int], optional): A tuple of actions indexed by player. (Default value = (2, 2))
        num_objectives (int, optional): The number of objectives in the game. (Default value = 2)
        mean (float, optional): The mean of the normal distribution. (Default value = 0)
        std (float, optional): The standard deviation of the normal distribution. (Default value = 1)
        rho (float, optional): The covariance between the players. (Default value = 0)
        rng (Generator, optional): A random number generator. (Default value = None)

    Returns:
        List[ndarray]: A list of payoff matrices representing the MONFG.

    """
    rng = rng if rng is not None else np.random.default_rng()
    num_players = len(player_actions)
    mean_arr = np.full(num_players, mean)
    std_arr = np.full(num_players, std)
    cov_matrix = np.full((num_players, num_players), rho)
    np.fill_diagonal(cov_matrix, std_arr)
    payoffs_shape = player_actions + tuple([num_objectives])  # Define the shape of the payoff matrices.
    all_payoffs = rng.multivariate_normal(mean_arr, cov_matrix, payoffs_shape)
    monfg = [all_payoffs[..., -1:].reshape(payoffs_shape)]  # Set payoffs for the first player
    for player in range(1, num_players):
        payoffs = all_payoffs[..., -player - 1:-player].reshape(payoffs_shape)
        monfg.append(payoffs)
    return monfg


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
    player_actions = get_player_actions(monfg)
    scalarised = []

    for payoff_matrix, u in zip(monfg, u_tpl):
        scalarised_payoffs = np.zeros(player_actions)

        for idx in np.ndindex(player_actions):  # Loop over all possible strategies.
            utility = u(payoff_matrix[idx])
            scalarised_payoffs[idx] = utility

        scalarised.append(scalarised_payoffs)
    return scalarised


def unique_ps_game(player_actions=(2, 2), num_objectives=2, reward_min_bound=0, reward_max_bound=5, rng=None):
    """Generate a random single-objective game where pure strategy payoffs for each opponent pure strategy are forced to
     be different.

     This generator makes a game where pure strategies have different payoffs for each opponent pure strategy. Note that
     this does not imply all payoffs are unique. The intention of this generator is to create a generic/non-degenerate
     game. In two-player two-action games this condition is sufficient. In all other games, this condition is necessary
     but not sufficient.

    Note:
        The current implementation samples a point, which is then hashed into a dictionary. When the dictionary reaches
        the correct length, there are enough unique payoffs.

    Args:
        player_actions (Tuple[int], optional): A tuple of actions indexed by player. (Default value = (2, 2))
        num_objectives (int, optional): The number of objectives in the game. (Default value = 2)
        reward_min_bound (int, optional): The minimum reward on an objective. (Default value = 0)
        reward_max_bound (int, optional): The maximum reward on an objective. (Default value = 5)
        rng (Generator, optional): A random number generator. (Default value = None)

    Returns:
        List[ndarray]: A game with unique pure strategy payoffs for each opponent pure strategy.

    Raises:
        Exception: When the generic game is impossible to create. This can happen when the action space of a player is
        larger than the reward space. This would imply the same reward for different actions, making the game
        non-generic.
    """
    if max(player_actions) > (reward_max_bound - reward_min_bound):
        raise Exception('The action space is larger than the reward space')

    rng = rng if rng is not None else np.random.default_rng()
    game = []
    game_shape = player_actions + (num_objectives,)

    for player, num_actions in enumerate(player_actions):  # Generate generic payoffs for each player.
        payoff_matrix = np.zeros(game_shape)
        opp_actions = player_actions[:player] + player_actions[player + 1:]

        for opp_strat in np.ndindex(opp_actions):  # Loop over opponent joint actions.
            payoffs_dict = {}

            while len(payoffs_dict) < num_actions:
                payoff = rng.integers(low=reward_min_bound, high=reward_max_bound, size=(1, num_objectives))
                payoff_str = np.array2string(payoff)
                payoffs_dict[payoff_str] = payoff

            for action, payoff in enumerate(payoffs_dict.values()):  # Insert the payoffs at the correct index.
                idx = opp_strat[:player] + (action,) + opp_strat[player:]
                payoff_matrix[idx] = payoff

        game.append(payoff_matrix)
    return game
