import numpy as np
import random
from numba import int32, float32
from numba.experimental import jitclass


spec = [
    ('rows', int32),
    ('cols', int32),
    ('values', float32[:, :])
]


@jitclass(spec)
class JitMatrix(object):
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.values = np.zeros((self.rows, self.cols), dtype=np.float32)

    def randomize(self, low, high):
        self.values = np.random.uniform(low, high, size=(self.rows, self.cols))
        return self

    def add(self, value):
        self.values = np.add(self.values, value)
        return self

    def multiply(self, value):
        self.values = np.multiply(self.values, value)
        return self

    def dot_product(self, jit_matrix):
        return np.dot(self.values, jit_matrix.values)

    def from_array(self, array_values):
        self.values = array_values.reshape((len(array_values), 1))
        return self

