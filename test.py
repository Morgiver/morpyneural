import numpy as np
import math
from Math.JitMatrix import JitMatrix
from Neural.JitLayerClass import JitLayer
from Neural.JitNeuralNetworkClass import JitNeuralNetwork

if __name__ == '__main__':
    nn = JitNeuralNetwork()
    nn.add_layer(2, 4, 'sigmoid')
    nn.add_layer(4, 2, 'sigmoid')

    print(nn.layers[0].weights.values)
