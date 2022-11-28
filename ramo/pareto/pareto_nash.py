import numpy as np

from ramo.pareto.verify import verify_pareto_nash
from ramo.strategy.operations import make_joint_strat_from_profile


def pure_strategy_pne(monfg):
    """Compute all Pure Strategy Pareto Nash Equilibria for a given MONFG.

    Args:
        monfg (MONFG): An MONFG object.

    Returns:
        List[List[ndarray]]: A list of pure joint strategies that are Pareto Nash equilibria.
    """
    pne_strats = []

    for joint_action in np.ndindex(monfg.player_actions):
        joint_strat = make_joint_strat_from_profile(joint_action, monfg.player_actions)

        if verify_pareto_nash(monfg, joint_strat):
            pne_strats.append(joint_strat)

    return pne_strats
