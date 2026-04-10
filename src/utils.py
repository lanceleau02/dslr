import sys

import pandas as pd


def print_error(msg):
    """
    Prints an error message to the console and exits the program.

    This function is used for immediate termination of the program execution
    when an error occurs. It displays the error message prefixed with the
    string "[Error]" to indicate the nature of the issue.

    :param msg: The error message to be printed.
    :type msg: str
    :return: None
    """
    print(f"[Error] {msg}")
    exit(1)


def check_args(args, length=1):
    """
    Checks the number of arguments provided against the expected length.

    :param args: The list of arguments provided to the program.
    :type args: list
    :param length: The expected number of arguments. Defaults to 1.
    :type length: int
    :return: None
    """

    if len(args) != length:
        print_error("Usage : python <scrip>.py <csv file>")


def get_abbreviation(name, max_len=12):
    """
    Génère un sigle pour un nom s'il dépasse une certaine longueur.
    """
    if len(name) <= max_len:
        return name
    words = name.split()
    if len(words) == 1:
        return name[:max_len]
    return "".join(word[0].upper() for word in words)


def check_file(file_name, drop_mv=False):
    """
    Checks if a given file can be read as a CSV file and handles possible
    errors.

    :param file_name: The path to the file that will be checked.
    :type file_name: str
    :param drop_mv: If True, rows with missing values will be dropped.
    :type drop_mv: bool
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

    if file is not None:
        required_columns = ["Index", "Hogwarts House"]
        missing_columns = [
            col for col in required_columns if col not in file.columns
        ]
        if missing_columns:
            print_error(f"Missing columns: {', '.join(missing_columns)}")

        if drop_mv:
            file = file.dropna(subset=required_columns)

        numeric_columns = file.select_dtypes(
            include=['float64', 'int64']).columns
        if len(numeric_columns) == 0 or (
                len(numeric_columns) == 1 and 'Index' in numeric_columns):
            print_error("No numeric columns found for analysis")

        if drop_mv:
            other_numeric = [col for col in numeric_columns if col != 'Index']
            file = file.dropna(subset=other_numeric, how='all')
            if file.empty:
                print_error("Dataset is empty after dropping NaN values")
    else:
        print_error("File could not be read")

    return file


def open_file(filename=None, drop_na=False):
    """
    Opens a file provided as a command-line argument, performs validation
    checks,
    and returns the file instance. This function ensures that command-line
    arguments
    and the file are properly validated before accessing the file.

    :param filename: Optional filename, defaults to reading from sys.argv.
    :param drop_na: If True, rows with missing values will be dropped.
    :raises ValueError: If the provided arguments are invalid or missing.
    :raises FileNotFoundError: If the specified file does not exist or is
        inaccessible.
    :raises OSError: For other file-related errors (e.g., permission issues).

    :return: A validated opened file object.
    :rtype: pandas.DataFrame
    """

    if not filename:
        args = sys.argv[1:]
        check_args(args)
        filename = args[0]

    file = check_file(filename, drop_mv=drop_na)

    return file
