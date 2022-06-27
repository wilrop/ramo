import json
from os.path import join

import pandas as pd


def save_metadata(path, metadata, name='metadata'):
    """Save keyword arguments as metadata in a JSON file.

    Args:
        path (str): The path to the directory in which all files will be saved.
        metadata (Dict): A dictionary of metadata to save.
        name (str, optional): The name of the metadata file. (Default value = metadata)
    """
    name = f'{name}.json'
    metadata_file = join(path, name)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)


def load_metadata(filepath):
    """Load metadata from a directory.

    Args:
        filepath (str): The path to the metadata file.

    Returns:
        Dict: A dictionary of metadata.
    """
    with open(filepath, 'w') as f:
        metadata = json.load(f)
    return metadata


def save_logs(logs, path, name, columns=None, index=False, mode='w'):
    """Save the logs to a CSV file.

    Args:
        logs (List[float]): A list of records.
        path (str): The path to the directory.
        name (str): The name of the logs file.
        columns (str, optional): The names of the columns.
        index (bool, optional): Whether to save the index as well.
        mode (str, optional): The Python write mode. (Default value = 'w')

    Returns:

    """
    filepath = join(path, f'{name}.csv')
    df = pd.DataFrame(logs, columns=columns)
    df.to_csv(filepath, index=index, mode=mode)
