from ramo.nash.IBR import iterated_best_response
from ramo.nash.fictitious_play import fictitious_play
from ramo.nash.moqups import moqups


def execute_algorithm(monfg, u_tpl, algorithm='MOQUPS', seed=None, options=None):
    """Execute the requested algorithm with the arguments provided by the user.

    Args:
        monfg (List[ndarray]): A list of payoff matrices representing the MONFG.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        algorithm (str, optional): The requested algorithm. (Default value = 'MOQUPS')
        seed (int, optional): Seed the NumPy generator. If set to None, the system seed is used. (Default value = None)
        options (Dict, optional): A dictionary of options for the selected algorithm. For a complete overview of the
            arguments per algorithm, see the algorithm documentation. (Default value = None)

    Returns:
        ndarray | Tuple[bool, List[ndarray]]: The return from the requested algorithm.

    Raises:
        Exception: When the requested algorithm is unknown.

    """
    if options is None:
        options = {}

    if algorithm == 'MOQUPS':
        results = moqups(monfg, u_tpl)
    elif algorithm == 'IBR':
        results = iterated_best_response(monfg, u_tpl, seed=seed, **options)
    elif algorithm == 'FP':
        results = fictitious_play(monfg, u_tpl, seed=seed, **options)
    else:
        raise Exception(f'The requested algorithm "{algorithm}" does not exist')
    return results
