import weakref
import DLSnow.functions as fun
from DLSnow.core import Parameter
import numpy as np


class Layer:
    def __init__(self):
        self._param = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._param.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.input = [weakref.ref(x) for x in inputs]
        self.output = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError

    def params(self):
        for name in self._param:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def clear_grad(self):
        for param in self.params():
            param.clear_grad()


class Linear(Layer):
    """
        auto generate weights and biases
    """
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype

        self.w = Parameter(None, name='w')
        if self.in_size is not None:
            self._init_w()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=self.dtype), name='b')

    def _init_w(self):
        INPUT, OUTPUT = self.in_size, self.out_size
        w_data = np.random.randn(INPUT, OUTPUT).astype(self.dtype) * np.sqrt(1. / INPUT)
        self.w.data = w_data

    def forward(self, x):
        if self.w.data is None:
            self.in_size = x.shape[1]
            self._init_w()

        var = fun.linear(x, self.w, self.b)
        return var
