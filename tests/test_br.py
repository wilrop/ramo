import unittest

import numpy as np

import ramo.best_response.best_response as br
import ramo.game.monfgs as monfgs
import ramo.utility_functions.functions as uf


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
        payoff_matrix = monfgs.monfg1[0]

        joint_strategy1 = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([1 / 3, 1 / 3, 1 / 3])]
        test1 = br.calc_expected_returns(player, payoff_matrix, joint_strategy1)
        correct1 = np.array([[3., 1.], [2., 2.], [1., 3.]])

        joint_strategy2 = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.25, 0.35, 0.4])]
        test2 = br.calc_expected_returns(player, payoff_matrix, joint_strategy2)
        correct2 = np.array([[2.85, 1.15], [1.85, 2.15], [0.85, 3.15]])

        np.testing.assert_almost_equal(test1, correct1)
        np.testing.assert_almost_equal(test2, correct2)

    def test_best_response(self):
        player = 1
        payoff_matrix = monfgs.monfg19[player]
        u = uf.u2

        joint_strategy = [np.array([0.5, 0.5]), np.array([0.75, 0.25])]
        test = br.calc_best_response(u, player, payoff_matrix, joint_strategy, global_opt=True)
        correct = np.array([0.75, 0.25])

        np.testing.assert_almost_equal(test, correct)


if __name__ == '__main__':
    unittest.main()
