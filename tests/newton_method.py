import numpy as np
from DLSnow import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


iters = 10
x = Variable(np.array(2.0))
for i in range(iters):
    print(i, x)
    y = f(x)
    x.clear_grad()
    y.backward(create_grad=True)

    x_grad = x.grad
    x.clear_grad()
    x_grad.backward()
    x_grad_2 = x.grad

    x.data -= x_grad.data / x_grad_2.data

