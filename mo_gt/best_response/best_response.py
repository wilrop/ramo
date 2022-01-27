import numpy as np
from scipy.optimize import minimize


def objective(strategy, expected_returns, u):
    """The objective function to minimise for is the negative SER, as we want to maximise for the SER.

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


def calc_expected_returns(player, payoff_matrix, joint_strategy):
    """Calculate the expected return for a player's actions with a given joint strategy.

    Args:
        player (int): The player to caculate expected returns for.
        payoff_matrix (ndarray): The payoff matrix for the given player.
        joint_strategy (List[ndarray]): A list of each player's individual strategy.

    Returns:
        ndarray: The expected returns for the given player's actions.

    """
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
        dim_array = np.ones((1, expected_returns.ndim), int).ravel()
        dim_array[opponent] = -1
        strategy_reshaped = strategy.reshape(dim_array)

        expected_returns = expected_returns * strategy_reshaped  # Calculate the probability of a joint state occurring.
        # We now take the sum of the weighted returns to get the expected returns.
        # We need keepdims=True to make sure that the opponent still exists at the correct axis, their action space is
        # just reduced to one action resulting in the expected return now.
        expected_returns = np.sum(expected_returns, axis=opponent, keepdims=True)

    expected_returns = expected_returns.reshape(num_actions, num_objectives)  # Cast the result to a correct shape.

    return expected_returns


def best_response(u, player, payoff_matrix, joint_strategy):
    """This function calculates a best response for a given player to a joint strategy.

    Args:
      u (callable): The utility function for this player.
      player (int): The player to caculate expected returns for.
      payoff_matrix (ndarray): The payoff matrix for the given player.
      joint_strategy (List[ndarray]): A list of each player's individual strategy.

    Returns:
      ndarray: A best response strategy.

    """
    expected_returns = calc_expected_returns(player, payoff_matrix, joint_strategy)
    num_actions = len(joint_strategy[player])

    unif_strategy = np.full(num_actions, 1 / num_actions)  # A uniform strategy as first guess for the optimiser.
    bounds = [(0, 1)] * num_actions  # Constrain probabilities to 0 and 1.
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Equality constraint is equal to zero by default.
    res = minimize(lambda x: objective(x, expected_returns, u), unif_strategy, bounds=bounds, constraints=constraints)

    br_strategy = res['x'] / np.sum(res['x'])  # In case of floating point errors.
    return br_strategy
