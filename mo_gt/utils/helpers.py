def array_slice(array, axis, start, end, step=1):
    """Slice an array across a desired index.

    Note:
        See: https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis

    Args:
        array (ndarray): An input array.
        axis (int): The axis to slice through.
        start (int): The start index of that axis.
        end (int): The end index of that axis.
        step (int): The step size of the slice.

    Returns:

    """
    return array[(slice(None),) * (axis % array.ndim) + (slice(start, end, step),)]
