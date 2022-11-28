import numpy as np

from ramo.nash.verify import verify_nash
from ramo.strategy.operations import make_joint_strat_from_profile


def mose(monfg, u_tpl):
    """Compute all Pure Strategy Nash Equilibria (PSNE) for a given MONFG.

    Note:
        MOSE, Multi-Objective Strategy Enumeration, is slow but guaranteed to be correct if using a global optimiser
        that has sufficient theoretical convergence guarantees for the given utility functions.

    Args:
        monfg (MONFG): An MONFG object.
        u_tpl (Tuple[callable]): A tuple of utility functions.

    Returns:
        List[List[ndarray]]: A list of pure joint strategies that are Nash equilibria.

    """
    psne_strats = []

    for joint_action in np.ndindex(monfg.player_actions):
        joint_strat = make_joint_strat_from_profile(joint_action, monfg.player_actions)

        if verify_nash(monfg, u_tpl, joint_strat):
            psne_strats.append(joint_strat)

    return psne_strats
