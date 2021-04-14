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

    def scale_multiply(self, value):
        """
        Multiply all values with a given value
        :param value:
        :return:
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.values[i, j] *= value

    def ew_add(self, matrix):
        """
        Element wise Adding operation with an other matrix
        :param matrix:
        :return:
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.values[i, j] = self.values[i, j] + matrix.values[i, j]

    def ew_multiply(self, matrix):
        """
        Element wise Multiply operation with an other matrix
        :param matrix:
        :return:
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.values[i, j] = self.values[i, j] * matrix.values[i, j]

    def dot_product(self, matrix):
        """
        Dot product between local values and an other matrix.
        Return a new Py2DMatrix
        :param matrix:
        :return:
        """
        new_matrix = Py2DMatrix(self.rows, matrix.cols)
        new_matrix.build()

        for i in range(new_matrix.cols):
            for j in range(new_matrix.cols):
                for k in range(self.cols):
                    new_matrix.values[i, j] += self.values[i, k] * matrix.values[k, j]

        return new_matrix
