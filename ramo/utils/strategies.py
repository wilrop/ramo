import numpy as np


def make_joint_strat_from_flat(flat_strat, player_actions):
    """Make a joint strategy from a flat joint strategy.

    Args:
        flat_strat (ndarray): A joint strategy as a flat array.
        player_actions (Tuple[int]): A tuple with the number of actions per player.

    Returns:
        List[ndarray]: A list of individual strategies.
    """
    curr = 0
    joint_strat = []

    for num_actions in player_actions:
        start = curr
        curr = curr + num_actions
        strat = flat_strat[start:curr]
        joint_strat.append(strat)

    return joint_strat


def normalise_strat(strat):
    """Normalise a strategy to sum to one.

    Args:
        strat (ndarray): A strategy array.

    Returns:
        ndarray: The same strategy as a probability vector.
    """
    total = np.sum(strat)
    if total > 0:
        norm_strat = strat / np.sum(strat)
    else:
        num_actions = len(strat)
        norm_strat = np.full(num_actions, 1 / num_actions)
    return norm_strat


def normalise_joint_strat(joint_strat):
    """Normalise all individual strategies in a joint strategy.

    Args:
        joint_strat (List[ndarray]): A list of individual strategies.

    Returns:
        List[ndarray]: A joint strategy with each individual strategy normalised.
    """
    norm_joint_strat = []

    for strat in joint_strat:
        norm_joint_strat.append(normalise_strat(strat))

    return norm_joint_strat
