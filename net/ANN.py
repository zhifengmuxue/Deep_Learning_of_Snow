import DLSnow.functions as F
import DLSnow.layers as L
from DLSnow import Model

"""
    Created on 2024-1-25
    @author: zfmx
    full connect network, also called MLP
    example:
        model = ANN((10,1))         two layer in 10 neural and 1 neural
        model = ANN((10, 20, 30, 1))      four layer 
"""


class ANN(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
