import unittest

import numpy as np

import ramo.game.monfgs as monfgs
import ramo.utility_functions.functions as uf
from ramo.commitment.execute_commitment import execute_commitment


class TestCommitment(unittest.TestCase):
    test_seed = 42

    def test_execute_commitment(self):
        game_str = 'game1'
        u_str = ('u1', 'u2')
        experiment = 'opt_comp_action'
        runs = 2
        episodes = 10
        rollouts = 10
        alternate = True

        game = monfgs.get_monfg(game_str)
        u_tpl = []
        for i in range(len(u_str)):
            u_tpl.append(uf.get_u(u_str[i]))
        u_tpl = tuple(u_tpl)

        data = execute_commitment(game, u_tpl, experiment=experiment, runs=runs, episodes=episodes, rollouts=rollouts,
                                  alternate=alternate, seed=self.test_seed)
        returns_log, action_probs_log, state_dist_log, com_probs_log, _ = data

        print(returns_log)
        print(action_probs_log)
        print(state_dist_log)
        print(com_probs_log)

        correct_returns_log = {
            0: [[0, 0, 8.080000000000002], [0, 1, 8.080000000000002], [0, 2, 8.5], [0, 3, 8.080000000000002],
                [0, 4, 8.0], [0, 5, 8.0], [0, 6, 8.5], [0, 7, 8.72], [0, 8, 8.18], [0, 9, 8.18], [1, 0, 8.18],
                [1, 1, 9.620000000000001], [1, 2, 8.18], [1, 3, 8.080000000000002], [1, 4, 8.0], [1, 5, 8.0],
                [1, 6, 8.0], [1, 7, 9.28], [1, 8, 8.98], [1, 9, 8.02]],
            1: [[0, 0, 3.9600000000000004], [0, 1, 3.9600000000000004], [0, 2, 3.75], [0, 3, 3.9600000000000004],
                [0, 4, 4.0], [0, 5, 4.0], [0, 6, 3.75], [0, 7, 3.6399999999999997], [0, 8, 3.9099999999999997],
                [0, 9, 3.9099999999999997], [1, 0, 3.9099999999999997], [1, 1, 3.19], [1, 2, 3.9099999999999997],
                [1, 3, 3.9600000000000004], [1, 4, 4.0], [1, 5, 4.0], [1, 6, 4.0], [1, 7, 3.36],
                [1, 8, 3.5100000000000002], [1, 9, 3.9899999999999998]]}
        correct_action_probs_log = {
            0: [[0, 0, 0.1, 0.3, 0.6], [0, 1, 0.3, 0.3, 0.4], [0, 2, 0.5, 0.3, 0.2], [0, 3, 0.3, 0.4, 0.3],
                [0, 4, 0.1, 0.7, 0.2], [0, 5, 0.6, 0.1, 0.3], [0, 6, 0.4, 0.2, 0.4], [0, 7, 0.4, 0.4, 0.2],
                [0, 8, 0.5, 0.2, 0.3], [0, 9, 0.4, 0.0, 0.6], [1, 0, 0.2, 0.6, 0.2], [1, 1, 0.0, 0.4, 0.6],
                [1, 2, 0.6, 0.0, 0.4], [1, 3, 0.3, 0.4, 0.3], [1, 4, 0.3, 0.6, 0.1], [1, 5, 0.2, 0.5, 0.3],
                [1, 6, 0.2, 0.6, 0.2], [1, 7, 0.1, 0.2, 0.7], [1, 8, 0.1, 0.4, 0.5], [1, 9, 0.3, 0.4, 0.3]],
            1: [[0, 0, 0.5, 0.3, 0.2], [0, 1, 0.3, 0.3, 0.4], [0, 2, 0.4, 0.4, 0.2], [0, 3, 0.4, 0.4, 0.2],
                [0, 4, 0.4, 0.3, 0.3], [0, 5, 0.3, 0.1, 0.6], [0, 6, 0.1, 0.3, 0.6], [0, 7, 0.6, 0.2, 0.2],
                [0, 8, 0.2, 0.7, 0.1], [0, 9, 0.2, 0.5, 0.3], [1, 0, 0.3, 0.1, 0.6], [1, 1, 0.2, 0.3, 0.5],
                [1, 2, 0.4, 0.3, 0.3], [1, 3, 0.4, 0.4, 0.2], [1, 4, 0.1, 0.6, 0.3], [1, 5, 0.5, 0.1, 0.4],
                [1, 6, 0.3, 0.4, 0.3], [1, 7, 0.2, 0.4, 0.4], [1, 8, 0.2, 0.3, 0.5], [1, 9, 0.3, 0.3, 0.4]]}
        correct_state_dist_log = np.array([[0.1, 0.6, 0.], [0., 0.1, 0.3], [0.4, 0.1, 0.4]])
        correct_com_probs_log = {
            0: [[0, 0, 0.6, 0.4], [0, 2, 0.3, 0.7], [0, 4, 0.5, 0.5], [0, 6, 0.4, 0.6], [0, 8, 0.6, 0.4],
                [1, 0, 0.3, 0.7], [1, 2, 0.7, 0.3], [1, 4, 0.4, 0.6], [1, 6, 0.7, 0.3], [1, 8, 0.6, 0.4]],
            1: [[0, 1, 0.4, 0.6], [0, 3, 0.5, 0.5], [0, 5, 0.5, 0.5], [0, 7, 0.3, 0.7], [0, 9, 0.6, 0.4],
                [1, 1, 0.3, 0.7], [1, 3, 0.4, 0.6], [1, 5, 0.4, 0.6], [1, 7, 0.5, 0.5], [1, 9, 0.3, 0.7]]}

        for player in range(len(game)):
            for returns, correct_returns in zip(returns_log[player], correct_returns_log[player]):
                np.testing.assert_almost_equal(returns, correct_returns)
            for action_probs, correct_action_probs in zip(action_probs_log[player], correct_action_probs_log[player]):
                np.testing.assert_almost_equal(action_probs, correct_action_probs)
            for com_probs, correct_com_probs in zip(com_probs_log[player], correct_com_probs_log[player]):
                np.testing.assert_almost_equal(com_probs, correct_com_probs)

        np.testing.assert_almost_equal(state_dist_log, correct_state_dist_log)


if __name__ == '__main__':
    unittest.main()
