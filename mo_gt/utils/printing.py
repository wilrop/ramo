from rich import print, box
from rich.console import Console
from rich.table import Table


def print_monfg(game_str, game):
    """Print an MONFG as a matrix.

    Args:
        game_str (str): The name of the game.
        game (List[ndarray]): The payoffs of the game.

    Returns:

    """
    console = Console()

    if len(game) == 2:
        table = Table(title=game_str, show_header=False, show_lines=True, box=box.HEAVY)
        for row1, row2 in zip(game[0], game[1]):
            row_data = []
            for payoff1, payoff2 in zip(row1, row2):
                row_data.append(f'{tuple(payoff1)}; {tuple(payoff2)}')
            table.add_row(*row_data)
        console.print(table)
    else:
        print(f'Printing games with more than two players is currently not supported')


def print_psne(game, psne_lst):
    """Pretty print a list of PSNE.

    Args:
      game (ndarray): A payoff matrix.
      psne_lst (List[ndarray]): A list of PSNE.

    Returns:

    """
    print('There are a total of ' + repr(len(psne_lst)) + ' pure strategy Nash equilibria')

    for idx, psne in enumerate(psne_lst):
        print(f'PSNE {idx} indexes: {psne}')

    if len(game.shape[:-1]) == 2:
        player_actions = game.shape[:-1]
        table = Table(title="MONFG", show_header=False, show_lines=True, box=box.HEAVY)

        for i in range(player_actions[0]):
            row_data = []

            for j in range(player_actions[1]):
                data = f'({i}, {j})'
                for indexes in psne_lst:
                    if indexes[0] == i and indexes[1] == j:
                        data = f'[black on green]({i}, {j})'
                        break

                row_data.append(data)
            table.add_row(*row_data)

        console = Console()
        console.print(table)


def print_ne(ne, joint_strategy):
    """Pretty print a Nash equilibrium

    Args:
      ne (bool): Whether the joint strategy is a Nash equilibrium.
      joint_strategy (List[ndarray]: The joint strategy that is a Nash equilibrium.

    Returns:

    """
    if ne:
        printable_strat = list(map(list, joint_strategy))
        print(f'The Nash equilibrium that was found is the joint strategy: {printable_strat}')
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
