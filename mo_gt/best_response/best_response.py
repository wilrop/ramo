import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize


def objective(strategy, expected_returns, u):
    """The objective function in an MONFG under SER.

    Implements the objective function for a player in an MONFG under SER. In a best-response calculation, players aim to
    maximise their utility. However, most often in optimisation we are given minimisers (e.g. SciPy's minimise).
    The sign of the utility is flipped as the objective to minimise, which effectively maximises the utility.

    Args:
      strategy (ndarray) The current estimate for the best response strategy.
      expected_returns (ndarray): The expected returns given all other players' strategies.
      u (callable): The utility function of this agent.

    Returns:
      float: The value on the objective with the provided arguments.

    """
    expected_vec = strategy @ expected_returns  # The expected vector of the strategy applied to the expected returns.
    objective = - u(expected_vec)  # The negative utility.
    return objective


def optimise_policy(expected_returns, u, init_strat=None):
    """Calculate a policy maximising a given utility function.

    Args:
        expected_returns (ndarray): The expected returns from the player's actions.
        u (callable): The player's utility function.
        init_strat (ndarray, optional): An initial guess for the optimal policy. (Default = None)

    Returns:

    """
    num_actions = len(expected_returns)

    if init_strat is None:
        init_strat = np.full(num_actions, 1 / num_actions)  # A uniform strategy as first guess for the optimiser.

    init_strats = [init_strat]
    for i in range(num_actions):
        pure_strat = np.zeros(num_actions)
        pure_strat[i] = 1
        init_strats.append(pure_strat)  # A pure strategy as initial guess for the optimiser.

    br_strategy = None
    br_utility = float('inf')
    for strat in init_strats:
        bounds = [(0, 1)] * num_actions  # Constrain probabilities to 0 and 1.
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Equality constraint is equal to zero by default.
        res = minimize(lambda x: objective(x, expected_returns, u), strat, bounds=bounds, constraints=constraints)

        if res['fun'] < br_utility:
            br_utility = res['fun']
            br_strategy = res['x'] / np.sum(res['x'])  # In case of floating point errors.

    return br_strategy


def calc_expected_returns(player, payoff_matrix, joint_strategy, api='np'):
    """Calculate the expected return for a player's actions with a given joint strategy.

    Args:
        player (int): The player to caculate expected returns for.
        payoff_matrix (ndarray): The payoff matrix for the given player.
        joint_strategy (List[ndarray]): A list of each player's individual strategy.

    Returns:
        ndarray: The expected returns for the given player's actions.

    """
    if api == 'np':
        api = np
    else:
        api = jnp

    num_objectives = payoff_matrix.shape[-1]
    num_actions = len(joint_strategy[player])
    num_players = len(joint_strategy)
    opponents = np.delete(np.arange(num_players), player)
    expected_returns = payoff_matrix

    for opponent in opponents:  # Loop over all opponent strategies.
        strategy = joint_strategy[opponent]  # Get this opponent's strategy.

        # We reshape this strategy to be able to multiply along the correct axis for weighting expected returns.
        # For example if you end up in [1, 2] or [2, 3] with 50% probability.
        # We calculate the individual expected returns first: [0.5, 1] or [1, 1.5]
        dim_array = api.ones((1, expected_returns.ndim), int).ravel()
        dim_array[opponent] = -1
        strategy_reshaped = strategy.reshape(dim_array)

        expected_returns = expected_returns * strategy_reshaped  # Calculate the probability of a joint state occurring.
        # We now take the sum of the weighted returns to get the expected returns.
        # We need keepdims=True to make sure that the opponent still exists at the correct axis, their action space is
        # just reduced to one action resulting in the expected return now.
        expected_returns = api.sum(expected_returns, axis=opponent, keepdims=True)

    expected_returns = expected_returns.reshape(num_actions, num_objectives)  # Cast the result to a correct shape.

    return expected_returns


def calc_best_response(u, player, payoff_matrix, joint_strategy, init_strat=None):
    """Calculate a best response for a given player to a joint strategy.

    Args:
      u (callable): The utility function for this player.
      player (int): The player to caculate expected returns for.
      payoff_matrix (ndarray): The payoff matrix for the given player.
      joint_strategy (List[ndarray]): A list of each player's individual strategy.
      init_strat (ndarray, optional): The initial guess for the best response. (Default = None)

    Returns:
      ndarray: A best response strategy.

    """
    expected_returns = calc_expected_returns(player, payoff_matrix, joint_strategy)
    br_strategy = optimise_policy(expected_returns, u, init_strat=init_strat)
    return br_strategy
