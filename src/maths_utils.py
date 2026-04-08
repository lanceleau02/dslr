def sum_(iterable, start=0):
    """
    Calculate the sum of a sequence of numbers.

    :param iterable: A sequence (list, tuple, or other iterable) of
        numeric values to be summed.
    :param start: An optional numeric value to be added to the total sum.
        Defaults to 0.
    :return: The numeric sum of the values in the iterable, including
        the optional starting value.
    :rtype: int or float
    """
    total = start
    for item in iterable:
        total += item
    return total


def sort_(array):
    """
    Sorts an array using a recursive implementation of the QuickSort algorithm.

    :param array: The list of elements to be sorted.
    :type array: list
    :return: A new sorted list containing elements from the input array.
    :rtype: list
    """
    if len(array) <= 1:
        return list(array)
    pivot = array[len(array) // 2]
    left = [x for x in array if x < pivot]
    middle = [x for x in array if x == pivot]
    right = [x for x in array if x > pivot]
    return sort_(left) + middle + sort_(right)


def mean_(array):
    """
    Calculate the arithmetic mean of a list of numbers.

    :param array: List of numerical values for which the mean needs to be
        calculated.
    :type array: list[float] | list[int]
    :return: The calculated mean of the input array, or None if the array
        is empty.
    :rtype: float | None
    """
    if len(array) == 0:
        return None
    n, total = 0, 0
    for value in array:
        n += 1
        total += value
    return total / n


def std_(array):
    """
    Calculate the standard deviation of a given array.

    :param array: A list of numerical values for which the standard deviation
        needs to be calculated.
    :type array: list[float | int]
    :return: The standard deviation of the input array or None if the array has
        fewer than two elements.
    :rtype: float | None
    """
    if len(array) == 0 or len(array) < 2:
        return None
    avg = mean_(array)
    squared_diff_sum = 0
    n = 0
    for value in array:
        squared_diff_sum += (value - avg) ** 2
        n += 1
    variance = squared_diff_sum / (n - 1)
    return variance ** 0.5


def min_(iterable):
    """
    Find the smallest element in the given iterable.

    :param iterable: An iterable object containing comparable elements.
    :return: The smallest element from the iterable.
    :rtype: Any
    :raises ValueError: If the given iterable is empty.
    """
    current_min = None
    for item in iterable:
        if current_min is None:
            current_min = item
        elif item < current_min:
            current_min = item
    if current_min is None:
        raise ValueError("min() arg is an empty sequence")
    return current_min


def max_(iterable):
    """
    Finds the maximum value in the provided iterable.

    :param iterable: An iterable containing comparable elements.
    :return: The maximum value found in the iterable.
    :rtype: Any
    :raises ValueError: If the provided iterable is empty.
    """
    current_max = None
    for item in iterable:
        if current_max is None:
            current_max = item
        elif item > current_max:
            current_max = item
    if current_max is None:
        raise ValueError("max() arg is an empty sequence")
    return current_max


def percentile_(sorted_array, percent):
    """
    Computes the percentile value from a sorted array.

    :param sorted_array: A sorted list of numerical values from which the
        percentile value is to be calculated. The array must be pre-sorted
        in ascending order. Providing an unsorted list will yield incorrect
        results.
    :type sorted_array: list[float]

    :param percent: A float value between 0 and 1, representing the
        percentile to compute as a decimal (e.g., 0.5 for the 50th percentile).
    :type percent: float

    :return: The computed percentile value as a float. If the input list is
        empty or if the percent value is out of range (less than 0 or
        greater than 1), the function returns None.
    :rtype: float or None
    """
    if not sorted_array:
        return None
    k = (len(sorted_array) - 1) * percent
    f = int(k)
    c = f + 1
    if c >= len(sorted_array):
        return sorted_array[f]
    weight = k - f
    return sorted_array[f] + weight * (sorted_array[c] - sorted_array[f])
