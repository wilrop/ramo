def constant_u(k):
    """Create a utility function that always outputs a constant value k.

    Args:
        k (float): The constant output of the utility function.

    Returns:
        callable: The generated utility function.
    """

    def u(x):
        return k

    return u
