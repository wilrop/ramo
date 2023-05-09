import unittest

import numpy as np

from ramo.pareto.verify import fast_p_prune, arg_p_prune, fast_c_prune


class TestMOSE(unittest.TestCase):
    candidates = np.array([[1, 1, 1],
                           [1, 2, 0],
                           [4, 0, 9],
                           [0, 0, 0],
                           [1, 1.5, 0.5],
                           [0, 1, 0],
                           [0, 1, 5],
                           [4, 0, 9]])
    pf_rem = np.array([[1, 1, 1], [1, 2, 0], [4, 0, 9], [1, 1.5, 0.5], [0, 1, 5]])
    pf_keep = np.array([[1, 1, 1], [1, 2, 0], [4, 0, 9], [1, 1.5, 0.5], [0, 1, 5], [4, 0, 9]])
    ccs = np.array([[1, 1, 1], [1, 2, 0], [4, 0, 9], [0, 1, 5]])

    def test_arg_p_prune(self):
        indcs = arg_p_prune(self.candidates, remove_duplicates=True)
        res = self.candidates[indcs]
        np.testing.assert_equal(res, self.pf_rem)

        indcs = arg_p_prune(self.candidates, remove_duplicates=False)
        res = self.candidates[indcs]
        np.testing.assert_equal(res, self.pf_keep)

    def test_fast_p_prune(self):
        res = fast_p_prune(self.candidates, remove_duplicates=True)
        np.testing.assert_equal(res, self.pf_rem)

        res = fast_p_prune(self.candidates, remove_duplicates=False)
        np.testing.assert_equal(res, self.pf_keep)

    def test_fast_c_prune(self):
        res = fast_c_prune(self.candidates)
        np.testing.assert_equal(res, self.ccs)


if __name__ == '__main__':
    unittest.main()
