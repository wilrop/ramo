import unittest

import numpy as np

import ramo.nash.fictitious_play as fp
import ramo.utility_function.functions as uf
from ramo.game.example_games import get_monfg


class TestFictitiousPlay(unittest.TestCase):
    test_seed = 42

    def test_alternating(self):
        game_str = 'game14'
        u_str = ('u1', 'u2', 'u3')

        monfg = get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        is_ne, joint_strat, _ = fp.fictitious_play(monfg, u_tpl, variant='alternating', seed=self.test_seed)

        self.assertTrue(is_ne)
        correct = [np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])]
        for idx, strategy in enumerate(joint_strat):
            np.testing.assert_almost_equal(strategy, correct[idx])

    def test_simultaneous(self):
        game_str = 'game14'
        u_str = ('u1', 'u2', 'u3')

        monfg = get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        is_ne, joint_strat, _ = fp.fictitious_play(monfg, u_tpl, variant='simultaneous', seed=self.test_seed)

        self.assertTrue(is_ne)
        correct = [np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])]
        for idx, strategy in enumerate(joint_strat):
            np.testing.assert_almost_equal(strategy, correct[idx])


if __name__ == '__main__':
    unittest.main()
