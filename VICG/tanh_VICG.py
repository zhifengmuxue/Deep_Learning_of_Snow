import numpy as np
from DLSnow.VICGutils import plot_dot_graph
from DLSnow import Variable
import DLSnow.functions as F
"""
    Created on 2024-1-24
    @author: zfmx
    you can get tanh's computational graph by run this code.
    iters: Select the number of times you need to derive it
"""
iters = 1


x = Variable(np.array([1.0, 2.0]))
y = F.tanh(x)
y.backward(create_grad=True)
x.name = 'x'
y.name = 'y'
for i in range(iters):
    grad_x = x.grad
    x.clear_grad()
    grad_x.backward(create_grad=True)
    print(i, x.grad)

grad_x = x.grad
grad_x.name = 'grad_x_' + str(iters + 1)
plot_dot_graph(grad_x, verbose=False, to_file='tanh.png')


