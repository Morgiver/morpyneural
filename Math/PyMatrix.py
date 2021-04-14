import numpy as np


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
