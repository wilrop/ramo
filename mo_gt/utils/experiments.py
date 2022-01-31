import os


def create_game_path(content, experiment, game, opt_init, mkdir=False):
    """Create a new directory based on the given parameters.

    Args:
      content (str): The type of content this directory will hold. Most often this is either 'data' or 'plots'.
      experiment (str): The name of the experiment that is being performed.
      game (str): The game that is experimented on.
      opt_init (bool): Whether the experiment involves optimistic initialisation.
      mkdir (bool, optional): Whether to create the directory.

    Returns:
      str: The path that was created.

    """
    path = f'{content}/{experiment}/{game}'

    if opt_init:
        path += '/opt_init'
    else:
        path += '/zero_init'

    if mkdir:
        os.makedirs(path, exist_ok=True)

    return path


def array_slice(array, axis, start, end, step=1):
    """Slice an array across a desired index.

    Notes
    ------
    See: https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis

    Args:
        array (ndarray): An input array.
        axis (int): The axis to slice through.
        start (int): The start index of that axis.
        end (int): The end index of that axis.
        step (int): The step size of the slice.

    Returns:

    """
    return array[(slice(None),) * (axis % array.ndim) + (slice(start, end, step),)]
