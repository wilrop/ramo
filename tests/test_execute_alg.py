import unittest

import numpy as np

import ramo.best_response.execute_algorithm as ea
import ramo.game.monfgs as monfgs
import ramo.utility_function.functions as uf


class TestExecuteAlgorithm(unittest.TestCase):
    test_seed = 42

    def test_execute_MOQUPS(self):
        game_str = 'game15'
        u_str = ('u1', 'u5', 'u6')

        game = monfgs.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = ea.execute_algorithm(game, u_tpl, algorithm='MOQUPS', seed=self.test_seed)
        correct = [[np.array([1., 0., 0.]), np.array([0., 1.]), np.array([0., 1., 0.])],
                   [np.array([0., 1., 0.]), np.array([1., 0.]), np.array([0., 0., 1.])],
                   [np.array([0., 1., 0.]), np.array([0., 1.]), np.array([1., 0., 0.])]]
        for test_joint_strat, correct_joint_strat in zip(test, correct):
            for test_strat, correct_strat in zip(test_joint_strat, correct_joint_strat):
                np.testing.assert_array_equal(test_strat, correct_strat)

    def test_execute_fp(self):
        game_str = 'game14'
        u_str = ('u1', 'u2', 'u3')

        game = monfgs.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        options = {'variant': 'alternating'}
        is_ne, joint_strat = ea.execute_algorithm(game, u_tpl, algorithm='FP', seed=self.test_seed, options=options)

        self.assertTrue(is_ne)
        correct = [np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])]
        for test_strat, correct_strat in zip(joint_strat, correct):
            np.testing.assert_almost_equal(test_strat, correct_strat)

    def test_execute_ibr(self):
        game_str = 'game14'
        u_str = ('u1', 'u2', 'u3')

        game = monfgs.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        options = {'variant': 'alternating'}
        is_ne, joint_strat = ea.execute_algorithm(game, u_tpl, algorithm='IBR', seed=self.test_seed, options=options)

        self.assertTrue(is_ne)
        correct = [np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])]
        for test_strat, correct_strat in zip(joint_strat, correct):
            np.testing.assert_almost_equal(test_strat, correct_strat)


if __name__ == '__main__':
    unittest.main()
