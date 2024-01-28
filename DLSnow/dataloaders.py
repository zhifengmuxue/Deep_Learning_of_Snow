import math
import random
import numpy as np

"""
    Created on 2024-1-28
    @author: zfmx
    select mini part of Dataset
"""


class DataLoader:
    """
        param
            dataset: the Dataset class
            batch_size: the batch size of train for one time
            shuffle: whether change the order of the dataset
        functions
            reset: reset the dataset
            next: next part of the dataset
    """
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.index = None
        self.iteration = None
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(self.dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)
        self.gpu = gpu
        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()
