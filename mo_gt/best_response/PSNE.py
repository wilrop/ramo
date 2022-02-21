import argparse
import time

import numpy as np

import mo_gt.games.games as games
import mo_gt.games.utility_functions as uf
import mo_gt.utils.printing as pt


def reduce_monfg(monfg, u_tpl):
    """Reduce an MONFG to an NFG by scalarisation. This is also known as the trade-off game.

    Args:
      monfg (List[ndarray]): An MONFG as a list of payoff matrices.
      u_tpl (Tuple[callable]): A tuple of utility functions.

    Returns:
      List[ndarray]: The scalarised MONFG.

    """
    nfg = []  # Collect the payoff matrices for the NFG.
    for player, u in enumerate(u_tpl):
        scalarised_payoff = games.scalarise_matrix(monfg[player], u)  # Scalarise each payoff matrix.
        nfg.append(scalarised_payoff)
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
    num_strategies = np.prod(player_actions)  # The number of possible pure strategies.

    for player, payoffs in enumerate(nfg):
        best_response_matrix = np.zeros(player_actions, dtype=bool)  # Initialise a new boolean best response matrix.
        maxima = np.amax(nfg[player], axis=player)  # The payoffs of the best responses to all other players strategies.
        for i in range(num_strategies):  # Loop over all joint strategies.
            idx = np.unravel_index(i, player_actions)  # Get the correctly shaped index of this strategy.
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


def calc_all_psne(monfg, u_tpl):
    """Calculate all Pure Strategy Nash Equilibria (PSNE) for a given MONFG with quasiconvex utility functions.

    Note:
        This algorithm is only guaranteed to be correct when using quasiconvex utility functions.

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str, default='game1', help="which MONFG to play")
    parser.add_argument('-u', type=str, default=['u1', 'u5'], nargs='+', help="The utility functions to use.")
    parser.add_argument('--player_actions', type=int, nargs='+', default=[3, 3],
                        help='The number of actions per agent')
    parser.add_argument('--num_objectives', type=int, default=2, help="The number of objectives for the random MONFG")
    parser.add_argument('--lower_bound', type=int, default=0, help='The lower reward bound.')
    parser.add_argument('--upper_bound', type=int, default=5, help='The upper reward bound.')

    args = parser.parse_args()

    start = time.time()  # Start measuring the time.

    if args.game == 'random':
        player_actions = args.player_actions
        monfg = games.generate_random_monfg(player_actions, args.num_objectives, args.lower_bound, args.upper_bound)
    else:
        monfg = games.get_monfg(args.game)

    u_tpl = tuple([uf.get_u(u_str) for u_str in args.u])  # These must be quasiconvex to ensure correctness.

    psne_lst = calc_all_psne(monfg, u_tpl)
    pt.print_psne(monfg, psne_lst, name=args.game)

    end = time.time()
    elapsed_secs = (end - start)
    print(f'Seconds elapsed: {str(elapsed_secs)}')
