import unittest

import numpy as np

import mo_gt.best_response.PSNEQ as psne
import mo_gt.envs.monfgs.examples as games
import mo_gt.utility_functions.examples as uf


class TestPSNE(unittest.TestCase):

    def test_small_game(self):
        game_str = 'game5'
        u_str = ('u6', 'u7')

        game = games.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = psne.calc_all_psne(game, u_tpl)
        correct = np.array([[0, 0], [1, 1]])
        np.testing.assert_array_equal(test, correct)

    def test_medium_game(self):
        game_str = 'game14'
        u_str = ('u1', 'u5', 'u6')

        game = games.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = psne.calc_all_psne(game, u_tpl)
        correct = np.array([[0, 1, 1], [1, 0, 1]])
        np.testing.assert_array_equal(test, correct)

    def test_large_game(self):
        game_str = 'game15'
        u_str = ('u1', 'u5', 'u6')

        game = games.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        test = psne.calc_all_psne(game, u_tpl)
        correct = np.array([[0, 1, 1], [1, 0, 2], [1, 1, 0]])
        np.testing.assert_array_equal(test, correct)


if __name__ == '__main__':
    unittest.main()
