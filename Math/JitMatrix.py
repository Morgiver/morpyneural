import numpy as np
import random
from numba import jit, int32, float32
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

