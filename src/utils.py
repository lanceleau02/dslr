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
