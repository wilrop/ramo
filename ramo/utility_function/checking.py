from sympy import hessian, degree_list, total_degree


def is_linear(func):
    """Check whether a function is linear.

    Args:
        func (Function): A sympy function.

    Returns:
        bool: Whether the function is linear.
    """
    degree = total_degree(func)
    return degree <= 1


def is_multilinear(func):
    """Check whether a function is multilinear.

    Args:
        func (Function): A sympy function.

    Returns:
        bool: Whether the function is multilinear.
    """
    degrees = degree_list(func)
    return all([1 == degree for degree in degrees])


def is_convex(func):
    """Check whether a function is convex.

    Args:
        func (Function): A sympy function.

    Returns:
        bool: Whether the function is convex.
    """
    hessian_matrix = hessian(func, list(func.free_symbols))
    return hessian_matrix.is_positive_semidefinite


def is_concave(func):
    """Check whether a function is concave.

    Args:
        func (Function): A sympy function.

    Returns:
        bool: Whether the function is concave.
    """
    return is_convex(-func)


def is_strictly_convex(func):
    """Check whether a function is strictly convex.

    Args:
        func (Function): A sympy function.

    Returns:
        bool: Whether the function is strictly convex.
    """
    hessian_matrix = hessian(func, list(func.free_symbols))
    return hessian_matrix.is_positive_definite


def is_strictly_concave(func):
    """Check whether a function is strictly concave.

    Args:
        func (Function): A sympy function.

    Returns:
        bool: Whether the function is strictly concave.
    """
    return is_strictly_convex(-func)
