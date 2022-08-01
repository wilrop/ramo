import unittest

import numpy as np

import ramo.game.monfgs as monfgs
import ramo.utility_function.functions as uf
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

        correct_returns_log = {
            0: [[0, 0, 8.32], [0, 1, 8.32], [0, 2, 8.72], [0, 3, 8.18], [0, 4, 8.72], [0, 5, 8.18], [0, 6, 8.02],
                [0, 7, 8.18], [0, 8, 8.98], [0, 9, 8.02], [1, 0, 8.02], [1, 1, 8.0], [1, 2, 8.080000000000002],
                [1, 3, 8.32], [1, 4, 8.080000000000002], [1, 5, 8.18], [1, 6, 8.02], [1, 7, 8.0], [1, 8, 8.0],
                [1, 9, 8.18]],
            1: [[0, 0, 3.84], [0, 1, 3.84], [0, 2, 3.6399999999999997], [0, 3, 3.9099999999999997],
                [0, 4, 3.6399999999999997], [0, 5, 3.9099999999999997], [0, 6, 3.9899999999999998],
                [0, 7, 3.9099999999999997], [0, 8, 3.5100000000000002], [0, 9, 3.9899999999999998],
                [1, 0, 3.9899999999999998], [1, 1, 4.0], [1, 2, 3.9600000000000004], [1, 3, 3.84],
                [1, 4, 3.9600000000000004], [1, 5, 3.9099999999999997], [1, 6, 3.9899999999999998],
                [1, 7, 4.0], [1, 8, 4.0], [1, 9, 3.9099999999999997]]}
        correct_action_probs_log = {
            0: [[0, 0, 0.3, 0.4, 0.3], [0, 1, 0.4, 0.3, 0.3], [0, 2, 0.6, 0.2, 0.2], [0, 3, 0.4, 0.1, 0.5],
                [0, 4, 0.5, 0.3, 0.2], [0, 5, 0.3, 0.1, 0.6], [0, 6, 0.4, 0.2, 0.4], [0, 7, 0.6, 0.2, 0.2],
                [0, 8, 0.4, 0.4, 0.2], [0, 9, 0.4, 0.3, 0.3], [1, 0, 0.2, 0.6, 0.2], [1, 1, 0.1, 0.5, 0.4],
                [1, 2, 0.3, 0.1, 0.6], [1, 3, 0.2, 0.3, 0.5], [1, 4, 0.3, 0.2, 0.5], [1, 5, 0.4, 0.3, 0.3],
                [1, 6, 0.4, 0.3, 0.3], [1, 7, 0.2, 0.4, 0.4], [1, 8, 0.3, 0.4, 0.3], [1, 9, 0.3, 0.5, 0.2]],
            1: [[0, 0, 0.2, 0.2, 0.6], [0, 1, 0.5, 0.3, 0.2], [0, 2, 0.3, 0.6, 0.1], [0, 3, 0.1, 0.6, 0.3],
                [0, 4, 0.4, 0.5, 0.1], [0, 5, 0.4, 0.2, 0.4], [0, 6, 0.4, 0.3, 0.3], [0, 7, 0.3, 0.3, 0.4],
                [0, 8, 0.7, 0.1, 0.2], [0, 9, 0.4, 0.2, 0.4], [1, 0, 0.2, 0.5, 0.3], [1, 1, 0.5, 0.3, 0.2],
                [1, 2, 0.3, 0.5, 0.2], [1, 3, 0.8, 0.1, 0.1], [1, 4, 0.4, 0.2, 0.4], [1, 5, 0.3, 0.6, 0.1],
                [1, 6, 0.2, 0.6, 0.2], [1, 7, 0.5, 0.2, 0.3], [1, 8, 0.3, 0.4, 0.3], [1, 9, 0.1, 0.4, 0.5]]}
        correct_state_dist_log = np.array([[0.2, 0.3, 0.2], [0.1, 0.1, 0.6], [0.2, 0.2, 0.1]])
        correct_com_probs_log = {
            0: [[0, 0, 0.6, 0.4], [0, 2, 0.5, 0.5], [0, 4, 0.5, 0.5], [0, 6, 0.3, 0.7], [0, 8, 0.6, 0.4],
                [1, 0, 0.4, 0.6], [1, 2, 0.5, 0.5], [1, 4, 0.5, 0.5], [1, 6, 0.6, 0.4], [1, 8, 0.4, 0.6]],
            1: [[0, 1, 0.7, 0.3], [0, 3, 0.3, 0.7], [0, 5, 0.5, 0.5], [0, 7, 0.7, 0.3], [0, 9, 0.5, 0.5],
                [1, 1, 0.7, 0.3], [1, 3, 0.7, 0.3], [1, 5, 0.5, 0.5], [1, 7, 0.3, 0.7], [1, 9, 0.4, 0.6]]}

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
