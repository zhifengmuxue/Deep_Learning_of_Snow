import numpy as np
from matplotlib import pyplot as plt
import DLSnow.evaluation as EVA
import DLSnow.datasets
from DLSnow.VICGutils import plot_dot_graph
from DLSnow import Variable
import DLSnow.functions as F
from net.ANN import ANN

# # x = Variable(np.array([1, 2, 3, 4, 5, 6]))
# x0 = Variable(np.array(1.0))
# # x1 = Variable(np.array([10]))
# # y = x0 + x1
# # print(y)
# # x0 = Variable(np.random.randn(2,3))
# # x1 = Variable(np.random.randn(3,4))
# y = F.exp(x0)
# print(y)
#
# y.backward()
# print(x0.grad)
# print(x1.grad.shape)
# x0 = Variable(np.array([1, 2, 3]))
# x1 = Variable(np.array([10]))
# y = x0 + x1
# print(y)
#
# y.backward()
# print(x1.grad)
# model = ANN((10, 3))
x = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8,0.1, 0.1]])
t = np.array([1,2,0])
# y = model(x)
# loss = F.softmax_cross_entropy(y, t)
# print(loss)
# y.backward()
# print(x.grad)
acc = EVA.accuracy(x, t)
print(acc)