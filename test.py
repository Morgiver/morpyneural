import numpy as np
import math
from Math.JitMatrix import JitMatrix
from Neural.JitLayerClass import JitLayer

if __name__ == '__main__':
    layer = JitLayer(4, 2, 'sigmoid')
    print(layer.weights.values)
