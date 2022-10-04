import unittest

import numpy as np

import ramo.game.monfgs as monfgs
import ramo.utility_function.functions as uf
from ramo.best_response.mose import mose


class TestMOSE(unittest.TestCase):

    def test_game1(self):
        game_str = 'game5'
        u_str = ('u1', 'u2')

        game = monfgs.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = mose(game, u_tpl)
        correct = [[np.array([0., 1., 0.]), np.array([0., 1., 0.])]]
        for test_joint_strat, correct_joint_strat in zip(test, correct):
            for test_strat, correct_strat in zip(test_joint_strat, correct_joint_strat):
                np.testing.assert_array_equal(test_strat, correct_strat)

    def test_game2(self):
        game_str = 'game18'
        u_str = ('u1', 'u2')

        game = monfgs.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = mose(game, u_tpl)
        correct = [[np.array([1., 0., 0.]), np.array([1., 0., 0.])], [np.array([0., 1., 0.]), np.array([0., 1., 0.])],
                   [np.array([0., 0., 1.]), np.array([0., 0., 1.])]]
        for test_joint_strat, correct_joint_strat in zip(test, correct):
            for test_strat, correct_strat in zip(test_joint_strat, correct_joint_strat):
                np.testing.assert_array_equal(test_strat, correct_strat)


if __name__ == '__main__':
    unittest.main()
