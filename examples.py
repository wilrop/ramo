# Example 1: Solving a game
from ramo.game.monfgs import get_monfg
from ramo.utility_function.functions import get_u

game = get_monfg('game1')  # Get a predefined game.
u1, u2 = get_u('u1'), get_u('u1')  # Get a utility function.

import ramo.utils.printing as pt
pt.print_monfg(game, 'Game 1')

import ramo.best_response.execute_algorithm as ea
u_tpl = (u1, u2)
psne = ea.execute_algorithm(game, u_tpl, algorithm='MOQUPS')
print(psne)

from ramo.utils.strategies import make_profile_from_pure_joint_strat
action_profiles = [make_profile_from_pure_joint_strat(joint_strat) for joint_strat in psne]
pt.print_monfg(game, 'Game 1', highlight_cells=action_profiles)

ne_fp = ea.execute_algorithm(game, u_tpl, algorithm='FP')
ne_ibr = ea.execute_algorithm(game, u_tpl, algorithm='IBR')


# Example 2: Running baseline algorithms
from ramo.game.generators import random_monfg
from ramo.utility_function.functions import get_u

game = random_monfg(player_actions=(3, 3), num_objectives=2, reward_min_bound=0, reward_max_bound=5)
u1, u2 = get_u('u1'), get_u('u1')
u_tpl = (u1, u2)

from ramo.learner.execute_learner import execute_learner

experiment = 'indep_ac'  # Independent actor critic.
runs = 2
episodes = 10
rollouts = 10

data = execute_learner(game, u_tpl, experiment=experiment, runs=runs, episodes=episodes, rollouts=rollouts)
returns_log, action_probs_log, state_dist_log, metadata = data

from ramo.commitment.execute_commitment import execute_commitment

experiment = 'non_stationary'
runs = 2
episodes = 10
rollouts = 10
alternate = False

data = execute_commitment(game, u_tpl, experiment=experiment, runs=runs, episodes=episodes, rollouts=rollouts, alternate=alternate)
returns_log, action_probs_log, state_dist_log, com_probs_log, metadata = data

# Example 3: Hypothesis testing
def u1(vec):
    x, y = vec
    return x ** 2 + y

def u2(vec):
    x, y = vec
    return x ** 2 + x * y + y ** 2

u_tpl = (u1, u2)

from sympy.abc import x, y
from ramo.utility_function.checking import is_convex, is_strictly_convex

symb_u1 = x ** 2 + y
res1 = is_convex(symb_u1)
print(res1)

symb_u2 = x ** 2 + x * y + y ** 2
res2 = is_strictly_convex(symb_u2)
print(res2)

res3 = is_strictly_convex(symb_u1)
print(res3)

import numpy as np
from ramo.game.checking import is_degenerate_pure

game = [np.array([[(1, 2), (2, 1)],
                  [(1, 2), (1, 2)]], dtype=float),
        np.array([[(1, 2), (2, 1)],
                  [(2, 1), (1, 2)]], dtype=float)]
res = is_degenerate_pure(game)
print(res)

game = [np.array([[(1, 2), (2, 1)],
                  [(2, 1), (1, 2)]], dtype=float),
        np.array([[(1, 2), (2, 1)],
                  [(2, 1), (1, 2)]], dtype=float)]

res = is_degenerate_pure(game)
print(res)

from ramo.best_response.execute_algorithm import execute_algorithm

psne = execute_algorithm(game, u_tpl)
print(psne)

from ramo.utils.printing import print_monfg
from ramo.utils.strategies import make_profile_from_pure_joint_strat

action_profiles = [make_profile_from_pure_joint_strat(ne) for ne in psne]
print_monfg(game, 'Special Game', action_profiles)

strat1 = np.array([0.5, 0.5])
strat2 = np.array([0.5, 0.5])
joint_strat = [strat1, strat2]

from ramo.best_response.best_response import calc_expected_returns

exp1 = calc_expected_returns(0, game[0], joint_strat)
print(exp1)

exp2 = calc_expected_returns(1, game[1], joint_strat)
print(exp2)

from ramo.best_response.best_response import verify_nash

is_ne = verify_nash(game, u_tpl, joint_strat)
print(is_ne)