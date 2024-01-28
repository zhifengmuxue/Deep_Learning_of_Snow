import numpy as np
from DLSnow import Variable
from net.TLnet import TwoLayerNet as TLnet


x = Variable(np.random.randn(5, 10), name='x')
model = TLnet(100, 10)
model.plot(x,to_file='TLnet.png')