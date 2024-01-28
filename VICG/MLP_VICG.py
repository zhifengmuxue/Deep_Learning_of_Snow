import numpy as np
from DLSnow import Variable
from net.ANN import ANN


x = Variable(np.random.randn(5, 10), name='x')
model = ANN((100, 20, 40, 10, 1))
model.plot(x, to_file='MLP.png')
