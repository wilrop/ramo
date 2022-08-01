import unittest

from sympy.abc import x, y

from ramo.utility_function.checking import *


class TestUtilityFunctionChecking(unittest.TestCase):

    def test_is_linear(self):
        u1 = x + y
        u2 = x * y + y
        res1 = is_linear(u1)
        res2 = is_linear(u2)
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_is_multilinear(self):
        u1 = x * y + y
        u2 = x * y + y ** 2
        res1 = is_multilinear(u1)
        res2 = is_multilinear(u2)
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_is_convex(self):
        u1 = x ** 2 + y
        u2 = x * y
        res1 = is_convex(u1)
        res2 = is_convex(u2)
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_is_concave(self):
        u1 = -(x ** 2 + y)
        u2 = x * y
        res1 = is_concave(u1)
        res2 = is_concave(u2)
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_is_strictly_convex(self):
        u1 = x ** 2 + x * y + y ** 2
        u2 = x ** 2 + y
        res1 = is_strictly_convex(u1)
        res2 = is_strictly_convex(u2)
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_is_strictly_concave(self):
        u1 = -(x ** 2 + x * y + y ** 2)
        u2 = -(x ** 2 + y)
        res1 = is_strictly_concave(u1)
        res2 = is_strictly_concave(u2)
        self.assertTrue(res1)
        self.assertFalse(res2)


if __name__ == '__main__':
    unittest.main()
