import os

import numpy as np
from jax.nn import softmax


def create_game_path(content, experiment, game, parent_dir=None, mkdir=True):
    """Create a new directory based on the given parameters.

    Args:
      content (str): The type of content this directory will hold. Most often this is either 'data' or 'plots'.
      experiment (str): The name of the experiment that is being performed.
      game (str): The game that is experimented on.
      parent_dir (str, optional): Parent directory for data and plots. (Default = None)
      opt_init (bool): Whether the experiment involves optimistic initialisation.
      mkdir (bool, optional): Whether to create the directory. (Default = True)

    Returns:
      str: The path that was created.

    """
    if parent_dir is None:
        parent_dir = '.'
    path = f'{parent_dir}/{content}/{experiment}/{game}'
    if mkdir:
        os.makedirs(path, exist_ok=True)

    return path


def make_strat_from_action(action, num_actions):
    """Turn an action into a strategy representation.

    Args:
        action (int): An action.
        num_actions (int): The number of possible actions.

    Returns:
        ndarray: A pure strategy as a numpy array.

    """
    strat = np.zeros(num_actions)
    strat[action] = 1
    return strat


def make_joint_strat(player_id, player_strat, opp_strat):
    """Make joint strategy from the opponent strategy and player strategy.

    Args:
        player_id (int): The id of the player.
        player_strat (ndarray): The strategy of the player.
        opp_strat (List[ndarray]): A list of the strategies of all other players.

    Returns:
        List[ndarray]): A list of strategies with the player's strategy at the correct index.

    """
    opp_strat.insert(player_id, player_strat)
    return opp_strat


def softmax_policy(theta):
    """Take a softmax over an array of parameters.

    Args:
      theta (ndarray): An array of policy parameters.

    Returns:
      ndarray: A probability distribution over actions as a policy.

    """
    policy = np.asarray(softmax(theta), dtype=float)
    policy = policy / np.sum(policy)
    return policy
