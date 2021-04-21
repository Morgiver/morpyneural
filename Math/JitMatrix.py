import numpy as np
import random
import numba
from numba import jit, int32, float32, deferred_type
from numba.experimental import jitclass


@jit(float32(float32))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@jit(float32(float32))
def relu(x):
    return np.maximum(0, x)


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
        """
        Randomize values between low and high
        :param low:
        :param high:
        :return:
        """
        self.values = np.random.uniform(low, high, size=(self.rows, self.cols))
        return self

    def add(self, value):
        """
        Addition operation
        :param value:
        :return:
        """
        self.values = np.add(self.values, value)
        return self

    def multiply(self, value):
        """
        Multiply operation
        :param value:
        :return:
        """
        self.values = np.multiply(self.values, value)
        return self

    def dot_product(self, jit_matrix):
        """
        Dot Product operation
        :param jit_matrix:
        :return:
        """
        return np.dot(self.values, jit_matrix.values)

    def from_array(self, array_values):
        """
        Set values from a 1D array values
        :param array_values:
        :return:
        """
        self.values = array_values.reshape((len(array_values), 1))
        return self

    def crossover(self, jit_matrix):
        """
        Crossover the matrix with an other to create a new one
        :param jit_matrix:
        :return:
        """
        new_matrix = JitMatrix(self.rows, self.cols)

        for i in range(self.rows):
            for j in range(self.cols):
                if random.random() <= 0.5:
                    new_matrix.values[i, j] = self.values[i, j]
                else:
                    new_matrix.values[i, j] = jit_matrix.values[i, j]

        return new_matrix

    def mutation(self, rate=0.001, low=-1, high=1):
        """
        For every values, will mutate it if the random value is smaller than the rate.
        The new value will be a random between low and high.
        :param rate:
        :param low:
        :param high:
        :return:
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if random.random() <= rate:
                    self.values[i, j] = random.uniform(low, high)

        return self

    def activate(self, fn_name):
        """
        Pass every values in an Activation Function
        :param fn_name:
        :return:
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if fn_name == "relu":
                    self.values[i, j] = relu(self.values[i, j])
                elif fn_name == "sigmoid":
                    self.values[i, j] = sigmoid(self.values[i, j])
                else:
                    self.values[i, j] = relu(self.values[i, j])

        return self


"""
Defining JitMatrix Types
"""
JitMatrixType = deferred_type()
JitMatrixType.define(JitMatrix.class_type.instance_type)
JitMatrixListType = numba.types.List(JitMatrix.class_type.instance_type, reflected=True)
