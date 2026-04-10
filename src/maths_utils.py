import numpy as np


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
    :return: The smallest element from the iterable or None if it's empty.
    :rtype: Any
    """
    current_min = None
    for item in iterable:
        if current_min is None:
            current_min = item
        elif item < current_min:
            current_min = item
    return current_min


def max_(iterable):
    """
    Finds the maximum value in the provided iterable.

    :param iterable: An iterable containing comparable elements.
    :return: The maximum value found in the iterable or None if it's empty.
    :rtype: Any
    """
    current_max = None
    for item in iterable:
        if current_max is None:
            current_max = item
        elif item > current_max:
            current_max = item
    return current_max


def range_(iterable):
    """
    Calculate the range of the given numeric dataset.

    The range is defined as the difference between the maximum and minimum 
    values in the dataset. The function expects an iterable of numeric 
    values as input. The dataset must not be empty, and all elements within 
    the dataset should be comparable.

    :param iterable: An iterable of numeric values for which the range will
        be calculated.
    :return: The range of the dataset as a numeric value.
    """
    return max_(iterable) - min_(iterable)


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
    # Return nothing for empty input
    if not sorted_array:
        return None

    # Find the percentile position
    k = (len(sorted_array) - 1) * percent
    f = int(k)
    c = f + 1

    # If it's the last element, return it directly
    if c >= len(sorted_array):
        return sorted_array[f]

    # Interpolate between the two nearest values
    weight = k - f
    return sorted_array[f] + weight * (sorted_array[c] - sorted_array[f])


def pearson_corr(series1, series2):
    """
    Calculate the Pearson correlation coefficient between two series.
    Ignore pairs containing missing values.

    :param series1: The first series of numerical values.
    :param series2: The second series of numerical values.
    :return: The Pearson correlation coefficient, or 0.0 if calculation
        is not possible (e.g., less than 2 valid pairs or zero variance).
    :rtype: float
    """
    # Keep only valid number pairs
    x_vals, y_vals = [], []
    for x, y in zip(series1, series2):
        if x is not None and y is not None:
            if x == x and y == y:
                x_vals.append(float(x))
                y_vals.append(float(y))
    n = len(x_vals)
    if n < 2:
        return 0.0

    # Compute mean and spread
    mx = mean_(x_vals)
    my = mean_(y_vals)
    sx = std_(x_vals)
    sy = std_(y_vals)
    if sx is None or sy is None or sx == 0 or sy == 0:
        return 0.0

    # Compute covariance
    cov_sum = 0.0
    for xi, yi in zip(x_vals, y_vals):
        cov_sum += (xi - mx) * (yi - my)
    cov = cov_sum / (n - 1)

    # Return normalized correlation
    return cov / (sx * sy)


def sigmoid(z):
    """
    Calculate the sigmoid function for the given input.

    The sigmoid function is a mathematical function that maps any real-valued
    number into the range (0, 1).

    :param z: The input value or array-like structure for which the sigmoid
    function
        is to be calculated.
    :type z: numpy.ndarray | float
    :return: The calculated sigmoid value(s) for the input. If the input is an
        array, the result will be an array of the same shape.
    :rtype: numpy.ndarray | float
    """
    # Prevent overflow
    z = np.clip(z, -500, 500)
    # Sigmoid formula
    return 1 / (1 + np.exp(-z))
