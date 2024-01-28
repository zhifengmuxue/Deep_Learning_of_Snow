from DLSnow.layers import Layer
from DLSnow import VICGutils
import DLSnow.functions as F
import DLSnow.layers as L


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        var = self.forward(*inputs)
        return VICGutils.plot_dot_graph(var, verbose=True, to_file=to_file)


class MLP(Model):
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