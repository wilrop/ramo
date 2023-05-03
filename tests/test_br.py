import unittest

import numpy as np

import ramo.strategy.best_response as br
import ramo.utility_function.functions as uf
from ramo.nash.verify import verify_nash
from ramo.game.example_games import get_monfg


class TestBestResponse(unittest.TestCase):

    def test_objective(self):
        unif_strat = np.array([1 / 3, 1 / 3, 1 / 3])
        pure_strat = np.array([0, 1, 0])
        random_strat = np.array([0.24, 0.29, 0.47])

        expected_returns = np.array([[1, 2], [3, 4], [5, 6]])
        u = uf.u5
        self.assertEqual(br.objective(unif_strat, expected_returns, u), 37)
        self.assertEqual(br.objective(pure_strat, expected_returns, u), 37)
        self.assertEqual(br.objective(random_strat, expected_returns, u), 47.294799999999995)

    def test_calc_expected_returns(self):
        player = 0
        monfg = get_monfg('game1')

        joint_strategy1 = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([1 / 3, 1 / 3, 1 / 3])]
        test1 = br.calc_expected_returns(player, monfg.payoffs[0], joint_strategy1)
        correct1 = np.array([[3., 1.], [2., 2.], [1., 3.]])

        joint_strategy2 = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.25, 0.35, 0.4])]
        test2 = br.calc_expected_returns(player, monfg.payoffs[0], joint_strategy2)
        correct2 = np.array([[2.85, 1.15], [1.85, 2.15], [0.85, 3.15]])

        np.testing.assert_almost_equal(test1, correct1)
        np.testing.assert_almost_equal(test2, correct2)

    def test_best_response(self):
        player = 1
        monfg = get_monfg('game19')
        u = uf.u2

        joint_strategy = [np.array([0.5, 0.5]), np.array([0.75, 0.25])]
        test = br.calc_best_response(u, player, monfg.payoffs[player], joint_strategy, global_opt=True)
        correct = np.array([0.75, 0.25])

        np.testing.assert_almost_equal(test, correct)

    def test_verify(self):
        monfg = get_monfg('game19')
        u1 = uf.u2
        u2 = uf.u2
        joint_strat = [np.array([0.5, 0.5]), np.array([0.75, 0.25])]

        res = verify_nash(monfg, [u1, u2], joint_strat)
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()
