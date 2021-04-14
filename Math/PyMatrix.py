import numpy as np
import random


class Py2DMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.values = None

    def build(self, dtype=np.float32):
        """
        Building new Matrix
        :param dtype:
        :return:
        """
        matrix = []
        for i in range(self.rows):
            cols_array = []
            for j in range(self.cols):
                cols_array.append(0)

            matrix.append(cols_array)

        self.values = np.array(matrix, dtype=dtype)

    def randomize(self, a=0, b=1):
        """
        Randomize all values between the two value a and b
        :param a:
        :param b:
        :return:
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.values[i, j] = random.uniform(a, b)

    def scale_add(self, value):
        """
        Add a value to all values
        :param value:
        :return:
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.values[i, j] += value
