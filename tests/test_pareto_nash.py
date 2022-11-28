import unittest

import numpy as np

from ramo.game.example_games import get_monfg
from ramo.pareto.pareto_nash import pure_strategy_pne


class TestMOSE(unittest.TestCase):

    def test_game1(self):
        game_str = 'game1'
        monfg = get_monfg(game_str)
        test = pure_strategy_pne(monfg)
        correct = [[np.array([1., 0., 0.]), np.array([1., 0., 0.])], [np.array([1., 0., 0.]), np.array([0., 1., 0.])],
                   [np.array([1., 0., 0.]), np.array([0., 0., 1.])], [np.array([0., 1., 0.]), np.array([1., 0., 0.])],
                   [np.array([0., 1., 0.]), np.array([0., 1., 0.])], [np.array([0., 1., 0.]), np.array([0., 0., 1.])],
                   [np.array([0., 0., 1.]), np.array([1., 0., 0.])], [np.array([0., 0., 1.]), np.array([0., 1., 0.])],
                   [np.array([0., 0., 1.]), np.array([0., 0., 1.])]]

        for test_joint_strat, correct_joint_strat in zip(test, correct):
            for test_strat, correct_strat in zip(test_joint_strat, correct_joint_strat):
                np.testing.assert_array_equal(test_strat, correct_strat)

    def test_game2(self):
        game_str = 'game5'
        game = get_monfg(game_str)
        test = pure_strategy_pne(game)
        correct = [[np.array([1., 0., 0.]), np.array([1., 0., 0.])], [np.array([0., 1., 0.]), np.array([0., 1., 0.])],
                   [np.array([0., 0., 1.]), np.array([0., 0., 1.])]]

        for test_joint_strat, correct_joint_strat in zip(test, correct):
            for test_strat, correct_strat in zip(test_joint_strat, correct_joint_strat):
                np.testing.assert_array_equal(test_strat, correct_strat)


if __name__ == '__main__':
    unittest.main()
