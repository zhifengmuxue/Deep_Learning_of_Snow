from DLSnow import as_variable, as_array
from DLSnow import Variable

"""
    Created on Mon
    @author: zfmx
    evaluation function for Deep Learning of Snow
"""


def accuracy(y_pred, target):
    y_pred, target = as_variable(y_pred), as_variable(target)

    pred = y_pred.data.argmax(axis=1).reshape(target.shape)
    result = (pred == target.data)
    acc = result.mean()
    return Variable(as_array(acc))

