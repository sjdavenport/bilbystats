"""
Functions to read in the data for bilbystats
"""

from importlib import resources
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import os


def read_data_path(data_name):
    """
    Retrieve the file path for a given data file name.

    Depending on the value of `data_name`, this function constructs the 
    appropriate file path from the `bilbystats.data` or 
    `bilbystats.data.prompts` module.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    data_name : str
        The name of the data file to locate. If it is 
        'simple_sentiment_prompt.txt', the function looks inside 
        'bilbystats.data.prompts'; otherwise, it defaults to 'bilbystats.data'.

    ---------------------------------------------------------------------------
    OUTPUT:
    path : str
        The resolved file path as a string.
    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    if data_name == 'simple_sentiment_prompt.txt':
        path = str(resources.files(
            "bilbystats.data.prompts").joinpath("simple_sentiment_prompt.txt"))
    elif data_name == 'sentiment_classification.txt':
        path = str(resources.files(
            "bilbystats.data.prompts").joinpath("sentiment_classification.txt"))
    else:
        path = str(resources.files("bilbystats.data").joinpath(data_name))

    return path


def read_data(data_name):
    """
    Load data from a specified file based on its extension.

    This function reads a data file given its name by first resolving its path
    using `read_data_path`. It supports reading `.txt`, `.csv`, and `.parquet` 
    files, and returns the contents accordingly.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    data_name : str
        The name of the data file to read. The file extension must be one of 
        '.txt', '.csv', or '.parquet'.

    ---------------------------------------------------------------------------
    OUTPUT:
    data : str or pandas.DataFrame
        The contents of the file. Returns a string if the file is a `.txt`,
        and a pandas DataFrame if it is a `.csv` or `.parquet`.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    path = read_data_path(data_name)

    file_ending = Path(path).suffix
    if file_ending == '.txt':
        with open(path, 'r') as file:
            data = file.read()
    elif file_ending == '.csv':
        data = pd.read_csv(path)
    elif file_ending == '.parquet':
        data = pd.read_parquet(path)

    return data


def read_txt_file(filepath, as_lines=False, strip_lines=True, encoding="utf-8"):
    """
    Reads a .txt file and returns its contents.

    Parameters:
        filepath (str): Path to the text file.
        as_lines (bool): If True, returns a list of lines. If False, returns a full string.
        strip_lines (bool): If True and as_lines=True, strips whitespace from each line.
        encoding (str): Encoding used to open the file (default: 'utf-8').

    Returns:
        str or list[str]: File contents as a string or list of lines.
    """
    with open(filepath, "r", encoding=encoding) as file:
        if as_lines:
            lines = file.readlines()
            if strip_lines:
                return [line.strip() for line in lines]
            return lines
        else:
            return file.read()


def check_dir(addon=''):
    try:
        env_path = resources.files(
            "bilbystats.defaults").joinpath("local_defaults.env")

        # Load environment variables from the .env file
        load_dotenv(dotenv_path=env_path)

        checkpoint_path = os.getenv("MODEL_CHECKPOINTS_DIR") + addon
    except:
        # If a local checkpoint path is not set defaults to the current directory.

        checkpoint_path = './' + addon

    return checkpoint_path
