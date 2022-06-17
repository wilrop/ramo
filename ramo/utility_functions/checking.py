from sympy import hessian, degree_list, total_degree


def is_linear(func):
    degree = total_degree(func)
    return degree <= 1


def is_multilinear(func):
    degrees = degree_list(func)
    return all([1 == degree for degree in degrees])


def is_convex(func):
    hessian_matrix = hessian(func, list(func.free_symbols))
    return hessian_matrix.is_positive_semidefinite


def is_concave(func):
    return is_convex(-func)


def is_strictly_convex(func):
    hessian_matrix = hessian(func, list(func.free_symbols))
    return hessian_matrix.is_positive_definite


def is_strictly_concave(func):
    return is_strictly_convex(-func)
