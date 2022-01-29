import unittest

import numpy as np

import mo_gt.best_response.execute_algorithm as ea
import mo_gt.games.games as games
import mo_gt.games.utility_functions as uf


class TestExecuteAlgorithm(unittest.TestCase):
    seed = 42

    def test_execute_PSNE(self):
        game_str = 'game15'
        u_str = ('u1', 'u5', 'u6')

        game = games.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = ea.execute_algorithm(game, u_tpl, algorithm='PSNE', seed=self.seed)
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

        test = ea.execute_algorithm(game, u_tpl, algorithm='FP', variant='alternating', seed=self.seed)
        self.assertTrue(test[0])
        print(test)
        correct = [np.array([1., 0.]), np.array([1., 0.]), np.array([1., 0.])]
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

        test = ea.execute_algorithm(game, u_tpl, algorithm='IBR', variant='alternating', seed=self.seed)
        self.assertTrue(test[0])
        correct = [np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])]
        for idx, strategy in enumerate(test[1]):
            np.testing.assert_array_equal(np.round(strategy, decimals=0), correct[idx])


if __name__ == '__main__':
    unittest.main()