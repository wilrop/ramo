import time

from mo_gt.best_response.IBR import iterated_best_response
from mo_gt.best_response.PSNE import calc_all_psne
from mo_gt.best_response.fictitious_play import fictitious_play


def execute_algorithm(monfg, u_tpl, algorithm='PSNE', seed=None, options=None):
    """Execute the requested algorithm with the arguments provided by the user.

    Args:
        monfg (List[ndarray]): A list of payoff matrices representing the MONFG.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        algorithm (str, optional): The requested algorithm. (Default value = 'PSNE')
        seed (int, optional): Seed the NumPy generator. If set to None, the system seed is used. (Default = None)
        options (Dict, optional): A dictionary of options for the selected algorithm. For a complete overview of the
            arguments per algorithm, see the algorithm documentation. (Default value = None)

    Returns:
        ndarray | Tuple[bool, List[ndarray]]: The return from the requested algorithm.

    Raises:
        Exception: When the requested algorithm is unknown.

    """
    start = time.time()  # Start measuring the time.

    if algorithm == 'PSNE':
        psne_lst = calc_all_psne(monfg, u_tpl)
        results = psne_lst
    elif algorithm == 'IBR':
        ne, final_strategy = iterated_best_response(monfg, u_tpl, seed=seed, **options)
        results = ne, final_strategy
    elif algorithm == 'FP':
        ne, final_strategy = fictitious_play(monfg, u_tpl, seed=seed, **options)
        results = ne, final_strategy
    else:
        raise Exception(f'The requested algorithm "{algorithm}" does not exist')

    end = time.time()
    elapsed_secs = (end - start)
    print("Seconds elapsed: " + str(elapsed_secs))

    return results
