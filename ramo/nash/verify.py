from ramo.strategy.best_response import calc_expected_returns, objective, optimise_policy


def verify_nash(monfg, u_tpl, joint_strat, epsilon=0, tol=1e-12, strict=False):
    """Verify whether the joint strategy is a Nash equilibrium

    Args:
        monfg (List[ndarray]): A list of payoff matrices.
        u_tpl (Tuple[callable]): A utility function per player.
        joint_strat (List[ndarray]): The joint strategy to verify.
        epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria. (Default value = 0)
        tol (float, optional): The tolerance in the utility calculation. The default is set to the shgo default from
        SciPy. (Default value = 1e-12)
        strict (bool, optional): Whether to count unsuccessful optimisations as unverified and thus returning False.
        (Default value = False)

    Note:
        A Nash equilibrium occurs whenever all strategies are best-responses to each other. We specifically use a global
        optimiser in this function to ensure all strategies are really best-responses and not local optima. Be aware
        that finding a global optimum for a function is computationally expensive, so this function might take longer
        than expected.

    Returns:
        bool: Whether the given joint strategy is a Nash equilibrium.
    """
    for player, (payoffs, u, strat) in enumerate(zip(monfg, u_tpl, joint_strat)):
        expected_returns = calc_expected_returns(player, payoffs, joint_strat)
        utility_from_strat = objective(strat, expected_returns, u)
        success, br_strat, br_utility = optimise_policy(expected_returns, u, global_opt=True)
        if (not strict or success) and utility_from_strat + epsilon + tol < br_utility:
            return False
    return True


def verify_all_nash(monfg, u_tpl, joint_strats, epsilon=0, tol=1e-12):
    """Globally verify if each joint strategy in a list is a Nash equilibrium.

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        joint_strats (List[ndarray]): A list of joint strategies to check.
        epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria. (Default value = 0)
        tol (float, optional): The tolerance in the utility calculation. The default is set to the shgo default from
        SciPy. (Default value = 1e-12)

    Returns:
        bool: Whether the joint_strategies in the list are actually Nash equilibria.
    """
    for joint_strat in joint_strats:
        if not verify_nash(monfg, u_tpl, joint_strat, epsilon=epsilon, tol=tol):
            return False
    return True
