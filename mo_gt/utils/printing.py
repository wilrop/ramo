from rich import print


def print_psne(psne_lst):
    """Pretty print a list of PSNE.

    Args:
      psne_lst (List[ndarray]): A list of PSNE.

    Returns:

    """
    print('There are a total of ' + repr(len(psne_lst)) + ' pure strategy Nash equilibria')
    for psne in psne_lst:
        print(psne)


def print_ne(ne, joint_strategy):
    """Pretty print a Nash equilibrium

    Args:
      ne (bool): Whether the joint strategy is a Nash equilibrium.
      joint_strategy (List[ndarray]: The joint strategy that is a Nash equilibrium.

    Returns:

    """
    if ne:
        print(f'The Nash equilibrium that was found is the joint strategy {joint_strategy}')
    else:
        print(f'No Nash equilibrium was found.')


def print_all_ne(ne_lst):
    """Pretty print a list of Nash equilibria.

    Args:
      ne_lst (List[List[ndarray]]): A list of Nash equilibria.

    Returns:

    """
    print(f'There are a total of {len(ne_lst)} Nash equilibria')
    for ne in ne_lst:
        print(repr(ne))


def print_start(algorithm):
    """Pretty print the introduction to an algorithm.

    Args:
      algorithm (str): The name of the algorithm.

    Returns:

    """
    print(f'Executing the {algorithm} algorithm')
    print(f'-----------------------------------------------------')
