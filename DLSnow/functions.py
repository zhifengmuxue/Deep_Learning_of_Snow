from DLSnow.core import as_variable
from DLSnow import Function, utils
import numpy as np
from DLSnow.utils import sum_to

"""
    Created on 2024-1-23
    @author: zfmx
    this code store all the functions 
"""


##############################################
#            base functions                  #
##############################################
class Square(Function):
    """
        Square function
        y = x ^ 2
        dy = 2 * x * dx
    """

    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, grad_y):
        x = self.input[0].data
        grad_x = x * 2 * grad_y
        return grad_x


def square(x):
    return Square()(x)


class Exp(Function):
    """
           Exponential function
           y = exp(x)
           dy = exp(x) * dx
   """

    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, grad_y):
        y = self.output[0]()
        grad_x = grad_y * y
        return grad_x


def exp(x):
    return Exp()(x)


class Log(Function):
    """
        Logarithmic function
        y = log x
    """

    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.input
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


class Sin(Function):
    """
        Sinusoidal function
        y = sin(x)
        dy = cos(x) * dx
    """

    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, grad_y):
        x, = self.input
        grad_x = cos(x) * grad_y
        return grad_x


def sin(x):
    return Sin()(x)


class Cos(Function):
    """
        Cosine function
        y = cos(x)
        dy = -sin(x) * dx
    """

    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, grad_y):
        x, = self.input
        grad_x = -sin(x) * grad_y
        return grad_x


def cos(x):
    return Cos()(x)


class Tanh(Function):
    """
        Tanh function
        y = ( exp(x) - exp(-x)) / ( exp(x) + exp(-x))
        dy = ( 1 - y ^ 2 ) * dx
    """

    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, grad_y):
        y = self.output[0]()
        grad_x = grad_y * (1 - y ** 2)
        return grad_x


def tanh(x):
    return Tanh()(x)


class Sum(Function):
    """
        Sum function
        sum the tensor
    """

    def __init__(self, axis, keepdims):
        self.x_shape = None
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, grad_y):
        grad_y = utils.reshape_sum_backward(grad_y, self.x_shape, self.axis, self.keepdims)
        grad_x = broadcast_to(grad_y, self.x_shape)
        return grad_x


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class MatMul(Function):
    def forward(self, x, w):
        y = x.dot(w)
        return y

    def backward(self, grad_y):
        x, w = self.input
        grad_x = matmul(grad_y, w.T)
        grad_w = matmul(x.T, grad_y)
        return grad_x, grad_w


def matmul(x, w):
    return MatMul()(x, w)


class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.input[0]
        y = self.output[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


###########################################
#           loss functions                #
###########################################


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, grad_y):
        x0, x1 = self.input
        diff = x0 - x1
        grad_x0 = grad_y * diff * (2. / len(diff))
        grad_x1 = -grad_x0
        return grad_x0, grad_x1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


##############################################
#          forward functon                   #
##############################################


class Linear(Function):
    """
        Linear transformation
        x * w + b
    """

    def forward(self, x, w, b):
        y = x.dot(w)
        if b is not None:
            y += b
        return y

    def backward(self, grad_y):
        x, w, b = self.input
        grad_b = None if b.data is None else sum_to(grad_y, b.shape)
        grad_x = matmul(grad_y, w.T)
        grad_w = matmul(x.T, grad_y)
        return grad_x, grad_w, grad_b


def linear(x, w, b=None):
    return Linear()(x, w, b)


################################################
#            active functions                  #
################################################

class Sigmoid(Function):
    """
        active function
    """

    def forward(self, x):
        y = np.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy):
        y = self.output[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.input
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, grad_y):
        y = self.output[0]()
        grad_x = y * grad_y
        sum_grad_x = grad_x.sum(axis=self.axis, keepdims=True)
        grad_x -= y * sum_grad_x
        return grad_x


def softmax(x, axis=1):
    return Softmax(axis)(x)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.log_sum_exp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.input
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        # convert to one-hot
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


###############################################################
#                    utils  function                          #
###############################################################

class Reshape(Function):
    """
    Reshape function
    change the tensor for we need
    """

    def __init__(self, shape):
        self.x_shape = None
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, grad_y):
        return reshape(grad_y, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    """
        Transpose function
        transpose the tensor
        and realization all method in numpy, such as
            A B C D
            1 2 3 4
        x.transpose(2, 1, 3, 4)
            B A C D
            2 1 3 4
    """

    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, grad_y):
        if self.axes is None:
            return transpose(grad_y)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(grad_y, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class Broadcast(Function):
    """
        Broadcast function
        broadcast the values
    """

    def __init__(self, shape):
        self.x_shape = None
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, grad_y):
        grad_x = sum_to(grad_y, self.x_shape)
        return grad_x


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Broadcast(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.x_shape = None
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, grad_y):
        grad_x = broadcast_to(grad_y, self.x_shape)
        return grad_x


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class GetItem(Function):
    """
        get item from model
    """

    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        var = x[self.slices]
        return var

    def backward(self, grad_y):
        x, = self.input
        f = GetItemGrad(self.slices, x.shape)
        return f(grad_y)


def get_item(x, slices):
    return GetItem(slices)(x)


class GetItemGrad(Function):
    def __init__(self, slices, shape):
        self.slices = slices
        self.shape = shape

    def forward(self, grad_y):
        grad_x = np.zeros(self.shape)
        np.add.at(grad_x, self.slices, grad_y)
        return grad_x

    def backward(self, grad2_x):
        return get_item(grad2_x, self.slices)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.input
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)
