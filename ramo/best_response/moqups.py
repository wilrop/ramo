import numpy as np

from ramo.game.generators import scalarised_game


def reduce_monfg(monfg, u_tpl):
    """Reduce an MONFG to an NFG by scalarisation. This is also known as the trade-off game.

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.
        u_tpl (Tuple[callable]): A tuple of utility functions.

    Returns:
        List[ndarray]: The scalarised MONFG.

    """
    nfg = scalarised_game(monfg, u_tpl)
    return nfg


def calc_nfg_psne(nfg, player_actions):
    """Calculate the PSNE for an NFG using the "underlining" method.

    Args:
        nfg (List[ndarray]): An NFG as a list of payoff matrices.
        player_actions (Tuple[int]): A tuple of the amount of actions per player.

    Returns:
        ndarray: The pure joint-strategies that are a PSNE.

    """
    best_responses = []  # Collect the best response matrices.

    for player, payoffs in enumerate(nfg):
        best_response_matrix = np.zeros(player_actions, dtype=bool)  # Initialise a new boolean best response matrix.
        maxima = np.amax(nfg[player], axis=player)  # The payoffs of the best responses to all other players strategies.
        for idx in np.ndindex(player_actions):  # Loop over all joint strategies.
            opp_strat = list(idx)  # Turn the index into a list for the next operation.
            del opp_strat[player]  # Find the opponent strategy that corresponds with this joint strategy.
            opp_strat = tuple(opp_strat)  # Turn it back into a tuple.
            max = maxima[opp_strat]  # Get the payoff of the best response to this opponent strategy.
            if payoffs[idx] >= max:  # If the payoff of this joint strategy is equal or greater.
                best_response_matrix[idx] = True  # It is a best response to the opponent strategies.

        best_responses.append(best_response_matrix)

    nash_equilibria = np.ones(player_actions, dtype=bool)  # Initialise a new matrix holding the PSNE.
    for i in range(len(best_responses)):
        nash_equilibria = np.logical_and(nash_equilibria,
                                         best_responses[i])  # Best response to all best responses is a NE.

    psne = np.argwhere(nash_equilibria)  # Get the action profiles that result in these PSNE.
    return psne


def moqups(monfg, u_tpl):
    """Compute all Pure Strategy Nash Equilibria (PSNE) for a given MONFG with quasiconvex utility functions [1].

    Note:
        MOQUPS, Multi-Objective and Quasiconvex Utilities for Pure Strategies, is only guaranteed to be correct when
        using quasiconvex utility functions.

    References:
        .. [1] Willem Röpke, Diederik M. Roijers, Ann Nowé, & Roxana Rădulescu. (2021). On Nash Equilibria in
            Normal-Form Games With Vectorial Payoffs.

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.
        u_tpl (Tuple[callable]): A tuple of utility functions.

    Returns:
        ndarray: The pure joint-strategies that are a PSNE.

    """
    player_actions = monfg[0].shape[:-1]  # Get the number of actions available to each player.
    nfg = reduce_monfg(monfg, u_tpl)  # Reduce the MONFG to an NFG.
    psne_lst = calc_nfg_psne(nfg, player_actions)  # Calculate the PSNE from these payoff matrices.
    return psne_lst
