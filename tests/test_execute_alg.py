import unittest

import numpy as np

import ramo.best_response.execute_algorithm as ea
import ramo.game.monfgs as monfgs
import ramo.utility_functions.functions as uf


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
        correct = np.array([[0, 1, 1], [1, 0, 2], [1, 1, 0]])
        np.testing.assert_array_equal(test, correct)

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
        for idx, strategy in enumerate(joint_strat):
            np.testing.assert_almost_equal(strategy, correct[idx])

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
        for idx, strategy in enumerate(joint_strat):
            np.testing.assert_almost_equal(strategy, correct[idx])


if __name__ == '__main__':
    unittest.main()
