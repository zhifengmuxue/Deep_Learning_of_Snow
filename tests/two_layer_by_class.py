import numpy as np
import DLSnow.functions as F
from DLSnow import Layer
import DLSnow.layers as L
import matplotlib.pyplot as plt

# data set
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

model = Layer()
model.l1 = L.Linear(10)
model.l2 = L.Linear(1)


# predict
def predict(data):
    var = model.l1(data)
    var = F.sigmoid(var)
    var = model.l2(var)
    return var


# train
lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    model.clear_grad()
    loss.backward()
    for layer in model.params():
        layer.data -= lr * layer.grad.data

    if i % 1000 == 0:
        print(i, loss)

# Plot the data
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('The final data')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()
