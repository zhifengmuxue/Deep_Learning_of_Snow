from DLSnow import Variable
from DLSnow import functions as fun
import numpy as np
import matplotlib.pyplot as plt

# data set from sin
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.arctan(2 * np.pi * x) + np.random.rand(100, 1)

# param
INPUT, HIDDEN, OUTPUT = 1, 10, 1
w1 = Variable(0.01 * np.random.randn(INPUT, HIDDEN))
w2 = Variable(0.01 * np.random.randn(HIDDEN, OUTPUT))
b1 = Variable(np.zeros(HIDDEN))
b2 = Variable(np.zeros(OUTPUT))


# predict
def predict(data):
    var = fun.linear(data, w1, b1)
    var = fun.sigmoid(var)
    var = fun.linear(var, w2, b2)
    return var


# train
lr = 0.2
iters = 10000
loss_history = []
for i in range(iters):
    y_pred = predict(x)
    loss = fun.mean_squared_error(y, y_pred)
    loss_history.append(loss.data)
    w1.clear_grad()
    w2.clear_grad()
    b1.clear_grad()
    b2.clear_grad()
    loss.backward()

    w1.data -= lr * w1.grad.data
    b1.data -= lr * b1.grad.data
    w2.data -= lr * w2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)


# Plot the data
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('The final data')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()


# Plot the loss
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.plot(loss_history)
plt.show()
