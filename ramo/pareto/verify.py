import numpy as np
import scipy.optimize as scopt
from pulp import *

from ramo.strategy.best_response import calc_expected_returns


def in_hull(x, points):
    """Check whether a point is a convex combination of a set of points.

    Args:
        x (ndarray): The point to check.
        points (ndarray): An array of points.

    Returns:
        bool: Whether the point was in the convex hull.
    """
    n_points = len(points)  # The number of points.
    c = np.zeros(n_points)  # Make an array of zeros of this size as the objective to minimise.
    A = np.r_[points.T, np.ones((1, n_points))]  # Add row of ones such that the strategy sums to 1.
    b = np.r_[x, np.ones(1)]  # The strategy array.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=scopt.OptimizeWarning)  # Suppress full row rank warnings.
        lp = scopt.linprog(c, A_eq=A, b_eq=b)  # Check if we can find a convex combination by linear programming.
    return lp.success


def c_prune(candidates):
    """Create a convex coverage set from a set of candidate points.

    References:
        .. [1] Roijers, D. M., & Whiteson, S. (2017). Multi-objective decision making. 34, 129–129.
            https://doi.org/10.2200/S00765ED1V01Y201704AIM034

    candidates (Set[Tuple]): A set of vectors.

    Returns:
        Set[Tuple]: A convex coverage set.

    """
    num_objectives = len(list(candidates)[0])
    zero_vec = tuple(np.zeros(num_objectives))
    p_candidates = p_prune(candidates)
    ccs = {zero_vec}

    while p_candidates:
        vector = p_candidates.pop()  # Get an element from the Pareto coverage set.
        p_candidates.add(vector)  # Pop removes the element so add it back for now.

        weight = find_weight(vector, ccs)  # Get a weight for which this vector improves the CCS.

        if weight is None:  # If there is none, discard the vector.
            p_candidates.remove(vector)
        else:  # Otherwise add the best vector to the CCS.
            new_vector = max(p_candidates, key=lambda x: np.dot(x, weight))
            p_candidates.remove(new_vector)
            ccs.add(new_vector)

    ccs.remove(zero_vec)
    return ccs


def pareto_dominates(a, b):
    """Check if the vector in a Pareto dominates vector b.

    Args:
        a (ndarray): A numpy array.
        b (ndarray): A numpy array.

    Returns:
        bool: Whether vector a dominates vector b.
    """
    a = np.array(a)
    b = np.array(b)
    return np.all(a >= b) and np.any(a > b)


def p_prune(candidates):
    """Create a Pareto coverage set from a set of candidate points.

    References:
        .. [1] Roijers, D. M., & Whiteson, S. (2017). Multi-objective decision making. 34, 129–129.
            https://doi.org/10.2200/S00765ED1V01Y201704AIM034

    Args:
        candidates (Set[Tuple]): A set of vectors.

    Returns:
        Set[Tuple]: A Pareto coverage set.
    """
    pcs = set()
    while candidates:
        vector = candidates.pop()
        to_remove = set(vector)
        for alternative in candidates:
            if pareto_dominates(alternative, vector):
                vector = alternative
                to_remove.add(alternative)
        candidates.difference(to_remove)
        pcs.add(vector)
    return pcs


def find_weight(vector, candidates):
    """Find a weight for which a specific vector improves on a CCS [1].

    References:
        .. [1] Roijers, D. M., & Whiteson, S. (2017). Multi-objective decision making. 34, 129–129.
            https://doi.org/10.2200/S00765ED1V01Y201704AIM034

    Args:
        vector (Tuple): A payoff vector.
        candidates (Set[Tuple]): The current CCS.

    Returns:
        ndarray | None: A weight array if it found one, otherwise None.
    """
    candidates = list(candidates)
    num_objectives = len(candidates[0])

    problem = LpProblem('findWeight', LpMaximize)
    x = LpVariable('x')
    w = []

    for obj in range(num_objectives):  # Make weight decision variables.
        w.append(LpVariable(f'w{obj}', 0, 1))

    for candidate in candidates:  # Add the constraints on the improvement of w.
        diff = list(np.subtract(vector, candidate))
        problem += lpDot(w, diff) - x >= 0

    problem += lpSum(w) == 1  # Weights should sum to one.
    problem += x  # Add x as the objective to maximise.
    success = problem.solve(solver=PULP_CBC_CMD(msg=False))  # Solve the problem.
    x = problem.objective.value()  # Get the objective value.
    weight_vec = np.zeros(num_objectives)
    for var in problem.variables():  # Get the weight values.
        if var.name[0] == 'w':
            weight_idx = int(var.name[-1])
            weight_vec[weight_idx] = var.value()
    if success and x > 0:
        return weight_vec
    return None


def verify_pareto_nash(monfg, joint_strat):
    """Verify whether a joint strategy is a Pareto Nash equilibrium.

    Args:
        monfg (List[ndarray]): A list of payoff matrices.
        joint_strat (List[ndarray]: A list of strategy arrays.

    Returns:
        bool: Whether the joint strategy is a Pareto Nash equilibrium.
    """
    for player, (payoffs, strat) in enumerate(zip(monfg, joint_strat)):
        expected_returns = calc_expected_returns(player, payoffs, joint_strat)
        candidates = set([tuple(vec) for vec in expected_returns])  # Get the candidates for the CCS.
        ccs = np.array(list(c_prune(candidates)))  # Compute a CCS.
        expected_vec = np.dot(strat, expected_returns)
        if not in_hull(expected_vec, ccs):  # Check whether the expected vector is in the CCS.
            return False
    return True


def verify_all_pareto_nash(monfg, joint_strats):
    """Globally verify if each joint strategy in a list is a Nash equilibrium.

    Args:
        monfg (List[ndarray]): An MONFG as a list of payoff matrices.
        joint_strats (List[ndarray]): A list of joint strategies to check.

    Returns:
        bool: Whether the joint_strategies in the list are actually Pareto Nash equilibria.
    """
    for joint_strat in joint_strats:
        if not verify_pareto_nash(monfg, joint_strat):
            return False
    return True
