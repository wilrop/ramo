import unittest

import numpy as np

from ramo.game.generators import covariant_monfg


class TestMONFGGenerators(unittest.TestCase):
    test_seed = 42

    def test_covariant_monfg(self):
        player_actions = (10, 10)
        num_objectives = 2
        mean = 0
        std = 1
        rho = -1
        rng = np.random.default_rng(seed=self.test_seed)
        monfg = covariant_monfg(player_actions=player_actions, num_objectives=num_objectives, mean=mean, std=std,
                                cov=rho, rng=rng)
        arr = np.stack([payoffs.flatten() for payoffs in monfg.payoffs], axis=0)
        test_res = np.cov(arr)
        correct_res = np.array([[0.9775708, -0.9775708], [-0.9775708, 0.9775708]])
        np.testing.assert_almost_equal(test_res, correct_res)


if __name__ == '__main__':
    unittest.main()
