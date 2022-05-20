import unittest

import numpy as np

import ramo.best_response.execute_algorithm as ea
import ramo.envs.monfgs.examples as games
import ramo.utility_functions.examples as uf


class TestExecuteAlgorithm(unittest.TestCase):
    test_seed = 42

    def test_execute_PSNEQ(self):
        game_str = 'game15'
        u_str = ('u1', 'u5', 'u6')

        game = games.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = ea.execute_algorithm(game, u_tpl, algorithm='PSNEQ', seed=self.test_seed)
        correct = np.array([[0, 1, 1], [1, 0, 2], [1, 1, 0]])
        np.testing.assert_array_equal(test, correct)

    def test_execute_fp(self):
        game_str = 'game14'
        u_str = ('u1', 'u2', 'u3')

        game = games.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        options = {'variant': 'alternating'}
        test = ea.execute_algorithm(game, u_tpl, algorithm='FP', seed=self.test_seed, options=options)

        self.assertTrue(test[0])
        correct = [np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])]
        for idx, strategy in enumerate(test[1]):
            np.testing.assert_array_equal(np.round(strategy, decimals=0), correct[idx])

    def test_execute_ibr(self):
        game_str = 'game14'
        u_str = ('u1', 'u2', 'u3')

        game = games.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        options = {'variant': 'alternating'}
        test = ea.execute_algorithm(game, u_tpl, algorithm='IBR', seed=self.test_seed, options=options)

        self.assertTrue(test[0])
        correct = [np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])]
        for idx, strategy in enumerate(test[1]):
            np.testing.assert_array_equal(np.round(strategy, decimals=0), correct[idx])


if __name__ == '__main__':
    unittest.main()
