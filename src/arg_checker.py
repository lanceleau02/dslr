import sys

import pandas as pd

from utils import print_error


def check_args():
    """ check the arguments """

    args = sys.argv[1:]

    if len(args) != 1:
        print_error("Usage : python describe.py <csv file>")

    file_name = args[0]
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
