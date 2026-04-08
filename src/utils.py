def sum(iterable, start=0):
    total = start
    for item in iterable:
        total += item
    return total

def min():

def mean(array, axis: Optional[Literal[0, 1]] = None):
    """
    Calculate the Mean of the values in an array.
    :param array: array containing the values.
    :param axis: axis along which the average is calculated.
    :return: the Mean value.
    """

def print_error(msg):
    """
    Print the error message precede by [Error] then exit the program.

    :param msg:
    :return:
    """
    print(f"[Error] {msg}")
    exit(1)