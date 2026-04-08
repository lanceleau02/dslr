import sys

import pandas as pd

from src.utils import print_error, check_args


def check_file(file_name):
    """
    Checks if a given file can be read as a CSV file and handles possible
    errors.

    :param file_name: The path to the file that will be checked.
    :type file_name: str
    :return: The pandas DataFrame if the file is successfully read, or None
    if an
        error occurs during the file reading process.
    :rtype: pandas.DataFrame or None
    """

    file = None

    try:
        file = pd.read_csv(file_name)
    except FileNotFoundError:
        print_error("File not found")
    except pd.errors.EmptyDataError:
        print_error("File is empty")
    except pd.errors.ParserError:
        print_error("File is not a csv file")

    return file


def open_file(filename=None):
    """
    Opens a file provided as a command-line argument, performs validation
    checks,
    and returns the file instance. This function ensures that command-line
    arguments
    and the file are properly validated before accessing the file.

    :raises ValueError: If the provided arguments are invalid or missing.
    :raises FileNotFoundError: If the specified file does not exist or is
        inaccessible.
    :raises OSError: For other file-related errors (e.g., permission issues).

    :return: A validated opened file object.
    :rtype: file
    """

    if not filename:
        args = sys.argv[1:]
        check_args(args)
        filename = args[0]

    file = check_file(filename)

    return file
