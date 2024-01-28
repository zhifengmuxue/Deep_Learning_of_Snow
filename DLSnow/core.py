import contextlib
import weakref
import numpy as np
import DLSnow

"""
    Created on  2024-1-20
    @author: zfmx
    package name : Deep Learning of Snow.
    use the Define-by-Run method to define the Variable and Function.
    all the functions we will use in this project, each function can forward and backward
    Base Class:
        Config: used to set the config,set the mode between train and inference
        Variable: store all variables and grad, data only support ndarray
        Function: base class for all functions
    until function:
        as_variable: change the obj to variable
        as_array : solve the numpy type change problem
        use_config : change the config
        no_grad: change the mode to no_grad, no calculate gradient
        set_variable: init the method of Variable calculation in __init__.py
    math class:
        Add: y = x0 + x1
        Multiply: y = x0 * x1
        Neg: y = -x
        Sub: y = x0 - x1
        Div: y = x0 / x1
"""


class Config:
    """ Config
        used to set config
        enable_backprop : set the mode backprop or no backprop
    """
    enable_backprop = True


class Variable:
    """ Variable
        self parameter
            data : type numpy array, the data to be stored in
            grad : type numpy array, the gradient of the variable
            creator : type of Function, the creator of the variable
            func : type python List, store all functions
            generation: the generation of the func

        def functions
            set_creator : set the creator of the variable
            backward : make a backward pass of the whole data
            clear_grad : clear the gradient of the Variable
            shape: return the shape of Variable
            reshape: change the shape we need
            transpose : change the shape
    """
    __array_priority = 200  # priority

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not support in Variable Class'.format(type(data)))
        self.data = data
        self.grad = None
        self.name = name
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    def set_creator(self, func):
        self.generation += 1
        self.creator = func

    def backward(self, retain_grad=False, create_grad=False):
        # dy = 1
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(fun):
            if fun not in seen_set:
                funcs.append(fun)
                seen_set.add(fun)
                funcs.sort(key=lambda lam: lam.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            y_grad_list = [output().grad for output in f.output]

            with use_config('enable_backprop', create_grad):
                x_grad_tuple = f.backward(*y_grad_list)
                if not isinstance(x_grad_tuple, tuple):
                    x_grad_tuple = (x_grad_tuple,)

                for x, x_grad in zip(f.input, x_grad_tuple):
                    if x.grad is None:
                        x.grad = x_grad
                    else:
                        x.grad += x_grad

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.output:
                    y().grad = None  # y is weak ref

    def clear_grad(self):
        self.grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return DLSnow.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return DLSnow.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        return DLSnow.functions.sum(self, axis, keepdims)

    @property
    def T(self):
        return DLSnow.functions.transpose(self)

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'


class Parameter(Variable):
    """
    Parameter class for Layers
    """
    pass


class Function:
    """ Function
        self parameter
            inputs : type python Tuple, the input of the function
            outputs : type python List, the output of the function
            generation : type int , the generation of function

        def functions
            __call__ : set the parameters of the function, use weak ref to reclaiming space
            forward : make a forward pass of inputs
            backward : make a backward pass of the outputs, form gradient equals 1

            forward and backward should be extended, or it will "raise NotImplementedError()"
    """

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xList = [x.data for x in inputs]
        yTuple = self.forward(*xList)
        if not isinstance(yTuple, tuple):
            yTuple = (yTuple,)
        outputs = [Variable(as_array(y)) for y in yTuple]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.input = inputs
            self.output = [weakref.ref(output) for output in outputs]  # weak ref

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *x):
        raise NotImplementedError()

    def backward(self, grad_y):
        raise NotImplementedError()


###############################################################
#                       until functions                       #
###############################################################

def as_variable(obj):
    """
    :param obj: obj
    :return: Variable
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(obj):
    """
    :param obj: obj
    :return: ndarray
    """
    if np.isscalar(obj):
        return np.array(obj)
    return obj


@contextlib.contextmanager
def use_config(name, value):
    """
    :param name: string
    :param value: bool
    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    """
    :return: use_config()
    """
    return use_config('enable_backprop', False)


#####################################################
#             calculations function                 #
#####################################################

class Add(Function):
    """
    :param: x0,x1,grad_y : Variable
    :return: Variable
    """
    def __init__(self):
        self.x1_shape = None
        self.x0_shape = None

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, grad_y):
        grad_x0, grad_x1 = grad_y, grad_y
        if self.x0_shape != self.x1_shape:
            grad_x0 = DLSnow.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = DLSnow.functions.sum_to(grad_x1, self.x1_shape)
        return grad_x0, grad_x1


class Multiply(Function):
    """
        :param: x0,x1,grad_y : Variable
        :return: Variable
    """
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, grad_y):
        x0, x1 = self.input
        grad_x0 = grad_y * x1
        grad_x1 = grad_y * x0
        if x0.shape != x1.shape:  # for broadcast
            grad_x0 = DLSnow.functions.sum_to(grad_x0, x0.shape)
            grad_x1 = DLSnow.functions.sum_to(grad_x1, x1.shape)
        return grad_x0, grad_x1


class Neg(Function):
    """
    :param: x, grad_y : Variable
    :return: Variable
    """
    def forward(self, x):
        return -x

    def backward(self, grad_y):
        return -grad_y


class Sub(Function):
    """
    :param: x0, x1, grad_y : Variable
    :return: Variable
    """
    def __init__(self):
        self.x1_shape = None
        self.x0_shape = None

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, grad_y):
        grad_x0 = grad_y
        grad_x1 = -grad_y
        if self.x0_shape != self.x1_shape:
            grad_x0 = DLSnow.functions.sum_to(grad_x0, self.x0_shape)
            grad_x1 = DLSnow.functions.sum_to(grad_x1, self.x1_shape)
        return grad_x0, grad_x1


class Div(Function):
    """
    :param: x0, x1, grad_y : Variable
    :return: Variable
    """
    def forward(self, x0, x1):
        y = x0 / x1
        return

    def backward(self, grad_y):
        x0, x1 = self.input
        grad_x0 = grad_y / x0
        grad_x1 = grad_y * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            grad_x0 = DLSnow.functions.sum_to(grad_x0, x0.shape)
            grad_x1 = DLSnow.functions.sum_to(grad_x1, x1.shape)
        return grad_x0, grad_x1


class Pow(Function):
    """
    :param: c: int
    :param: x, grad_y: Variable
    :return: Variable
    """
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, grad_y):
        x = self.input[0]
        c = self.c
        grad_x = c * x ** (c - 1) * grad_y
        return grad_x


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_array(x1)
    return Multiply()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def r_sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def r_div(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def r_pow(x, c):
    return Pow(c)(x)


def setup_variable():
    """
    Sets up the calculate in python method
    """
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = r_sub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = r_div
    Variable.__pow__ = r_pow
    Variable.__getitem__ = DLSnow.functions.get_item
    Variable.max = DLSnow.functions.max
    Variable.min = DLSnow.functions.min
