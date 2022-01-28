import time

import numpy as np
import mo_gt.utils.printing as pt

from mo_gt.best_response.PSNE import calc_all_psne
from mo_gt.best_response.IBR import iterated_best_response
from mo_gt.best_response.fictitious_play import fictitious_play


def execute_algorithm(monfg, u_tpl, algorithm='PSNE', variant='alternating', iterations=1000, seed=None):
    """Execute the requested algorithm with the arguments provided by the user.

    Args:
        monfg (List[ndarray]): A list of payoff matrices representing the MONFG.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        algorithm (str, optional): The requested algorithm. (Default value = 'PSNE')
        variant (str, optional): The variant to use when executing fictitious play or iterated best response.
            (Default value = 'alternating')
        iterations (int, optional): The number of iterations to execute. Only used when executing fictitious play or
            iterated best response. (Default value = 1000)
        seed (int, optional): Seed the NumPy generator. If set to None, the system seed is used. (Default = None)

    Returns:
        ndarray | Tuple[bool, List[ndarray]]: The return from the requested algorithm.

    Raises:
        Exception: When the requested algorithm is unknown.

    """
    start = time.time()  # Start measuring the time.

    if seed is not None:
        np.random.seed(seed)  # Set the numpy seed.

    if algorithm == 'PSNE':
        psne_lst = calc_all_psne(monfg, u_tpl)
        pt.print_psne(monfg[0], psne_lst)
        results = psne_lst
    elif algorithm == 'IBR':
        ne, final_strategy = iterated_best_response(monfg, u_tpl, max_iter=iterations, variant=variant)
        pt.print_ne(ne, final_strategy)
        results = ne, final_strategy
    elif algorithm == 'FP':
        ne, final_strategy = fictitious_play(monfg, u_tpl, max_iter=iterations, variant=variant)
        pt.print_ne(ne, final_strategy)
        results = ne, final_strategy
    else:
        raise Exception(f'The requested algorithm "{algorithm}" does not exist')

    end = time.time()
    elapsed_secs = (end - start)
    print("Seconds elapsed: " + str(elapsed_secs))

    return results
