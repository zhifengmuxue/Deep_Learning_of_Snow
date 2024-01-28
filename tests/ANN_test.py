import numpy as np
from net.ANN import ANN
from DLSnow.functions import mean_squared_error as Loss
import matplotlib.pyplot as plt
# data set
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
model = ANN((10, 20, 30, 40, 1))
# model = MLP((10, 20, 1))

# train
lr = 0.2
iters = 100000

for i in range(iters):
    y_pred = model.forward(x)
    loss = Loss(y, y_pred)
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
plt.title('The predicted values')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = model.forward(t)
plt.plot(t, y_pred.data, color='r')
plt.show()


