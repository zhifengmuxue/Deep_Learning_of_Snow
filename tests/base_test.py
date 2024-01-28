import unittest
from DLSnow.core import *

"""
    Created on 2024-1-21
    @author: zfmx
    this code used to test.dot the base class of this project
    include Variable class and Function class
"""


class VariableTest(unittest.TestCase):
    def test_init(self):
        data = np.array(2.0)
        var = Variable(data)
        self.assertEqual(var.data, data)
        self.assertIsNone(var.grad)
        self.assertIsNone(var.creator)

    def test_set_creator(self):
        var = Variable(np.array(2.0))
        func = Function()
        var.set_creator(func)
        self.assertEqual(var.creator, func)

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        y.backward()
        expected = np.array(4.0)
        self.assertEqual(x.grad, expected)


if __name__ == '__main__':
    unittest.main()
