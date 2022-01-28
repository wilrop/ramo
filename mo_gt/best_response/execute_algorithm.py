import time

import numpy as np

import mo_gt.games.games as games
from mo_gt.best_response.IBR import iterated_best_response
from mo_gt.best_response.fictitious_play import fictitious_play


def execute_algorithm(algorithm='PSNE', game='game1', u=('u1', 'u1'), player_actions=(5, 5), variant='alternating',
                      iterations=1000, num_objectives=2, lower_bound=0, upper_bound=5, seed=None):
    """Execute the requested algorithm with the arguments provided by the user.

    Args:
        algorithm:
        game:
        u:
        player_actions:
        variant:
        iterations:
        num_objectives:
        lower_bound:
        upper_bound:
        seed:

    Returns:

    """
    start = time.time()  # Start measuring the time.

    if args.seed is not None:
        np.random.seed(args.seed)  # Set the numpy seed.

    if args.game == 'random':  # We get the game
        player_actions = tuple(args.player_actions)
        monfg = games.generate_random_monfg(player_actions, args.num_objectives, args.lower_bound, args.upper_bound)
    else:
        monfg = games.get_monfg(args.game)

    player_actions = monfg[0].shape[:-1]  # Get the number of actions available to each player.
    u_tpl = tuple([games.get_u(u_str) for u_str in args.u])

    algorithm = args.algorithm
    variant = args.variant
    iterations = args.iterations

    if algorithm == 'PSNE':
        psne_lst = find_all_psne(monfg, player_actions, u_tpl)
        util.print_psne(psne_lst)
    elif algorithm == 'IBR':
        ne, final_strategy = iterated_best_response(u_tpl, player_actions, monfg, max_iter=iterations, variant=variant)
        util.print_ne(ne, final_strategy)
    elif algorithm == 'FP':
        ne, final_strategy = fictitious_play(u_tpl, player_actions, monfg, max_iter=iterations, variant=variant)
        util.print_ne(ne, final_strategy)
    else:
        raise Exception('The requested algorithm does not exist')

    end = time.time()
    elapsed_secs = (end - start)
    print("Seconds elapsed: " + str(elapsed_secs))
