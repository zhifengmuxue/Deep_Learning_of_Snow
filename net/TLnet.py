from DLSnow import Model
import DLSnow.layers as L
import DLSnow.functions as F


class TwoLayerNet(Model):
    def __init__(self, hidden_size, output_size, activation=F.sigmoid):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(output_size)
        self.active = activation

    def forward(self, x):
        y = self.active(self.l1(x))
        y = self.l2(y)
        return y


