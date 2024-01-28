import numpy as np
from net.ANN import ANN
from DLSnow.functions import mean_squared_error as Loss
from DLSnow.optimizers import Adam
import matplotlib.pyplot as plt
# data set
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = ANN((hidden_size, 1))
optimizer = Adam(lr)
optimizer.setup(model)

for i in range(max_iter):
    y_pred = model.forward(x)
    loss = Loss(y, y_pred)
    model.clear_grad()
    loss.backward()
    optimizer.update()
    if i % 1000 == 0:
        print('Epoch: ' + str(i) + ', loss: ' + str(loss))


