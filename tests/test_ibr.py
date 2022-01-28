import unittest

import numpy as np

import mo_gt.best_response.IBR as ibr
import mo_gt.games.games as games
import mo_gt.games.utility_functions as uf


class TestIteratedBestResponse(unittest.TestCase):

    def test_alternating(self):
        np.random.seed(42)

        game_str = 'game14'
        u_str = ('u1', 'u2', 'u3')

        game = games.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = ibr.iterated_best_response(game, u_tpl, variant='alternating')
        self.assertTrue(test[0])
        correct = [np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])]
        for idx, strategy in enumerate(test[1]):
            np.testing.assert_array_equal(np.round(strategy, decimals=0), correct[idx])

    def test_simultaneous(self):
        np.random.seed(42)

        game_str = 'game14'
        u_str = ('u1', 'u2', 'u3')

        game = games.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = ibr.iterated_best_response(game, u_tpl, variant='simultaneous')
        self.assertTrue(test[0])
        correct = [np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])]
        for idx, strategy in enumerate(test[1]):
            np.testing.assert_array_equal(np.round(strategy, decimals=0), correct[idx])


if __name__ == '__main__':
    unittest.main()
