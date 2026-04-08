def sum_(iterable, start=0):
    total = start
    for item in iterable:
        total += item
    return total

def sort_(array):
    if len(array) <= 1:
        return list(array)
    pivot = array[len(array) // 2]
    left = [x for x in array if x < pivot]
    middle = [x for x in array if x == pivot]
    right = [x for x in array if x > pivot]
    return sort_(left) + middle + sort_(right)

def mean_(array):
    if len(array) == 0:
        return None
    n, total = 0, 0
    for value in array:
        n += 1
        total += value
    return total / n

def std_(array):
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
    if not sorted_array:
        return None
    k = (len(sorted_array) - 1) * percent
    f = int(k)
    c = f + 1
    if c >= len(sorted_array):
        return sorted_array[f]
    weight = k - f
    return sorted_array[f] + weight * (sorted_array[c] - sorted_array[f])

def print_error(msg):
    """
    Print the error message precede by [Error] then exit the program.

    :param msg:
    :return:
    """
    print(f"[Error] {msg}")
    exit(1)