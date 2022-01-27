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
