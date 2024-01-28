from DLSnow.core import *
import unittest

"""
    Created on  2024-1-21
    Author: zfmx
    this code is used to test.dot the functions of the class
    numerical_diff function : functions used for numerical gradient calculations
    other Class test.dot : forward test.dot , backward test.dot , gradient check test.dot
"""


def numerical_diff(f, x, eps=1e-10):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class FunctionTest(unittest.TestCase):
    def test_call(self):
        x = Variable(np.array(2.0))
        f = Square()
        y = f(x)
        self.assertEqual(y.creator, f)
        self.assertEqual(f.input, (x,))
        self.assertEqual(f.output, [y])

    def test_forward(self):
        x = Variable(np.array(2.0))
        f = Square()
        y = f(x)
        expected = np.array(4.0)
        self.assertEquals(y.data, expected)

    def test_backward(self):
        f = Square()
        f.input = (Variable(np.array(2.0)),)
        var = f.backward(1)
        expected = np.array(4.0)
        self.assertEqual(var, expected)

    def test_add_self(self):
        x = Variable(np.array(1.0))
        y = add(x, x)
        y.backward()
        expected = np.array(2.0)
        self.assertEqual(x.grad, expected)

    def test_complex_diagrams_backward(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        expected = np.array(64.0)
        self.assertEqual(x.grad, expected)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flag = np.allclose(x.grad, num_grad)
        self.assertTrue(flag)


class ExponentialTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(1.0))
        y = exp(x)
        expected = np.exp(1.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = exp(x)
        y.backward()
        expected = np.exp(2.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()
        num_grad = numerical_diff(exp, x)
        flag = np.allclose(x.grad, num_grad)
        self.assertTrue(flag)


class AddTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(2.0))
        y = add(x0, x1)
        self.assertEqual(y.data, np.array(3.0))

    def test_backward(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(2.0))
        z = add(x, y)
        z.backward()
        y_expected = x_expected = np.array(1.0)
        self.assertEqual(x.grad, x_expected)
        self.assertEqual(y.grad, y_expected)


class MultiplyTest(unittest.TestCase):
    def test_backward(self):
        x0 = Variable(np.array(3.0))
        x1 = Variable(np.array(4.0))
        y = mul(x0, x1)
        y.backward()
        x0_expected = np.array(4.0)
        x1_expected = np.array(3.0)
        self.assertEqual(x0.grad, x0_expected)
        self.assertEqual(x1.grad, x1_expected)


if __name__ == '__main__':
    unittest.main()
