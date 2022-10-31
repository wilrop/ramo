import numpy as np

from ramo.game.properties import get_player_actions
from ramo.nash.verify import verify_nash
from ramo.strategy.operations import make_strat_from_action


def mose(monfg, u_tpl):
    """Compute all Pure Strategy Nash Equilibria (PSNE) for a given MONFG.

    Note:
        MOSE, Multi-Objective Strategy Enumeration, is slow but guaranteed to be correct if using a global optimiser
        that has sufficient theoretical convergence guarantees for the given utility functions.

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.
        u_tpl (Tuple[callable]): A tuple of utility functions.

    Returns:
        List[List[ndarray]]: A list of pure joint strategies that are Nash equilibria.

    """
    player_actions = get_player_actions(monfg)
    psne_strats = []

    for joint_action in np.ndindex(player_actions):
        joint_strat = []

        for action, num_actions in zip(joint_action, player_actions):
            strat = make_strat_from_action(action, num_actions)
            joint_strat.append(strat)

        if verify_nash(monfg, u_tpl, joint_strat):
            psne_strats.append(joint_strat)

    return psne_strats
