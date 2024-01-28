import math

import numpy as np
from DLSnow import Variable
from DLSnow.VICGutils import plot_dot_graph


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
        (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
# z = Variable(goldstein(x, y))
x.name = 'x'
y.name = 'y'
z.name = 'z'
z.backward()
plot_dot_graph(z, verbose=False, to_file='goldstein.png')


# x = Variable(np.array(np.pi/4))
# y = my_sin(x)
# y.backward()
# print(x.grad)
# print(y.data)
#
#
# plot_dot_graph(y, verbose=False, to_file='sin_1e-15.png')
#
#
# def my_sin(x, threshold=1e-15):
#     # Taylor's Formula
#     y = 0
#     for i in range(100000):
#         c = (-1) ** i / math.factorial(i * 2 + 1)
#         t = c * x ** (2 * i + 1)
#         y = y + t
#         if abs(t.data) < threshold:
#             break
#     return y