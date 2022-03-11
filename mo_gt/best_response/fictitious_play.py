import argparse
import time

import numpy as np

import mo_gt.games.monfg as games
import mo_gt.games.utility_functions as uf
import mo_gt.utils.printing as pt
from mo_gt.best_response.Player import FPPlayer
from mo_gt.best_response.best_response import verify_nash


def fictitious_play(monfg, u_tpl, epsilon=0, max_iter=1000, init_joint_strategy=None, variant='simultaneous',
                    global_opt=False, verify=True, seed=None):
    """Execute the fictitious play algorithm on a given MONFG and utility functions.

    There are two variants of the fictitious play algorithm implemented, simultaneous and alternating fictitious play.
    These variants are not equivalent in general. In the simultaneous variant, all players calculate their best-response
    strategy simultaneously. The alternating variant does it by alternating.

    Note:
        At this point in time, the algorithm does not find cycles and will continue to execute until the maximum number
        of iterations is reached.

    Args:
        monfg (List[ndarray]): A list of payoff matrices representing the MONFG.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        epsilon (float, optional): An optional parameter to allow for approximate Nash equilibria. (Default = 0)
        max_iter (int, optional): The maximum amount of iterations to run IBR for. (Default value = 1000)
        init_joint_strategy (List[ndarray], optional): Initial guess for the joint strategy. (Default value = None)
        variant (str, optional): The variant to use, which is either simultaneous or alternating.
        (Default value = 'simultaneous')
        global_opt (bool, optional): Whether to use a global optimiser or a local one. (Default = False)
        verify (bool, optional): Verify if a converged joint strategy is a Nash equilibrium. When set to true, this uses
        a global optimiser and might be computationally expensive. (Default = True)
        seed (int, optional): The initial seed for the random number generator. (Default value = None)

    Returns:
        Tuple[bool, List[ndarray]]: Whether or not we reached a Nash equilibrium and the final joint strategy.

    """
    rng = np.random.default_rng(seed=seed)

    def simultaneous_variant(joint_strategy):
        """Execute one iteration of the simultaneous fictitious play variant.

        Args:
            joint_strategy (List[ndarray]): The current joint strategy.

        Returns:
            Tuple[bool, List[ndarray]]: Whether the policies have converged and the new joint strategy.

        """
        converged = True
        actions = []

        for player in players:  # Collect actions.
            actions.append(player.select_action())

        for player in players:  # Update the empirical state distributions.
            player.update_empirical_strategies(actions)

        for player_id, player in enumerate(players):  # Update the policies simultaneously.
            done, br = player.update_strategy(epsilon=epsilon, global_opt=global_opt)
            joint_strategy[player_id] = br  # Update the joint strategy.
            if not done:
                converged = False

        return converged, joint_strategy

    def alternating_variant(joint_strategy):
        """Execute one iteration of the alternating fictitious play variant.

        Args:
            joint_strategy (List[ndarray]): The current joint strategy.

        Returns:
            Tuple[bool, List[ndarray]]: Whether the policies have converged and the new joint strategy.

        """
        converged = True

        for player_id, player in enumerate(players):  # Loop once over each player to update with alternating.
            actions = []

            for action_player in players:  # Collect actions.
                actions.append(action_player.select_action())

            for update_player in players:  # Update the empirical state distributions.
                update_player.update_empirical_strategies(actions)

            done, br = player.update_strategy(epsilon=epsilon,
                                              global_opt=global_opt)  # Update the current player's policy.
            joint_strategy[player_id] = br  # Update the joint strategy.
            if not done:
                converged = False

        return converged, joint_strategy

    pt.print_start('Fictitious Play')

    player_actions = monfg[0].shape[:-1]  # Get the number of actions available to each player.
    players = []  # A list to hold all the players.
    joint_strategy = []  # A list to hold the current joint strategy.

    for player_id, u in enumerate(u_tpl):  # Loop over all players to create a new FPAgent object.
        payoff_matrix = monfg[player_id]
        init_strategy = None
        if init_joint_strategy is not None:
            init_strategy = init_joint_strategy[player_id]
        player = FPPlayer(player_id, u, player_actions, payoff_matrix, init_strategy=init_strategy, rng=rng)
        players.append(player)
        joint_strategy.append(player.strategy)

    nash_equilibrium = False  # The current joint strategy is not known to be a Nash equilibrium at this point.

    if variant == 'simultaneous':
        execute_iteration = simultaneous_variant
    else:
        execute_iteration = alternating_variant

    for i in range(max_iter):
        print(f'Performing iteration {i}')
        converged, joint_strategy = execute_iteration(joint_strategy)

        if converged:  # If IBR converged, check if we can guarantee a Nash equilibrium.
            if global_opt:  # If we used a global optimiser, it is guaranteed to be a Nash equilibrium.
                nash_equilibrium = True
            elif verify:  # Otherwise check if the user wanted to verify.
                nash_equilibrium = verify_nash(monfg, u_tpl, joint_strategy, epsilon=epsilon)
            break

    return nash_equilibrium, joint_strategy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str, default='game1', help="which MONFG to play")
    parser.add_argument('--variant', type=str, default='alternating', choices=['simultaneous', 'alternating'])
    parser.add_argument('--iterations', type=int, default=1000, help="The maximum number of iterations.")
    parser.add_argument('-u', type=str, default=['u1', 'u2'], nargs='+', help="The utility functions to use.")
    parser.add_argument('--player_actions', type=int, nargs='+', default=[5, 5],
                        help='The number of actions per player')
    parser.add_argument('--num_objectives', type=int, default=2, help="The number of objectives for the random MONFG.")
    parser.add_argument('--lower_bound', type=int, default=0, help='The lower reward bound.')
    parser.add_argument('--upper_bound', type=int, default=5, help='The upper reward bound.')

    args = parser.parse_args()

    start = time.time()  # Start measuring the time.

    if args.game == 'random':
        player_actions = tuple(args.player_actions)
        monfg = games.generate_random_monfg(player_actions, args.num_objectives, args.lower_bound, args.upper_bound)
    else:
        monfg = games.get_monfg(args.game)

    u_tpl = tuple([uf.get_u(u_str) for u_str in args.u])
    variant = args.variant
    iterations = args.iterations

    ne, final_strategy = fictitious_play(monfg, u_tpl, max_iter=iterations, variant=variant)
    pt.print_ne(ne, final_strategy)

    end = time.time()
    elapsed_secs = (end - start)
    print("Seconds elapsed: " + str(elapsed_secs))
