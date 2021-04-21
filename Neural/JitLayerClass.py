import numba
from numba.experimental import jitclass
from numba import deferred_type
from Math.JitMatrix import JitMatrix, JitMatrixType


spec = [
    ('weights', JitMatrixType),
    ('biases', JitMatrixType),
    ('activation', str)
]


@jitclass(spec)
class JitLayer(object):
    def __init__(self):
        """
        Layer contain Weights, Biases and Activation data's of a Neural Network
        """
        self.weights = JitMatrix(1, 1)
        self.biases = JitMatrix(1, 1)
        self.activation = "sigmoid"

    def build(self, inputs, nodes, activation):
        """
        Building Weights and Biases matrices, define the Activation Function
        :param inputs:
        :param nodes:
        :param activation:
        :return:
        """
        # Weights PyD2Matrix
        self.weights = JitMatrix(nodes, inputs).randomize(-1, 1)

        # Biases PyD2Matrix
        self.biases = JitMatrix(nodes, 1).randomize(-1, 1)

        # Set Activation
        self.activation = activation

        return self

    def feed_forward(self, inputs):
        """
        Feed Forwarding inputs into Weights, Biases and Activation function
        :param inputs:
        :return:
        """
        return self.weights.dot_product(inputs).add(self.biases).activate(self.activation)

    def evolve(self, parent_a, parent_b, learning_rate):
        """
        Evolving Weights and Biases matrices values
        :param parent_a:
        :param parent_b:
        :param learning_rate:
        :return:
        """
        # Evolve Weights
        self.weights = parent_a.weights.crossover(parent_b.weights).mutation(learning_rate, -1, 1)

        # Evolve Biases
        self.biases = parent_a.biases.crossover(parent_b.biases).mutation(learning_rate, -1, 1)

        return self


"""
Define JitLayerTypes
"""
JitLayerType = deferred_type()
JitLayerType.define(JitLayer.class_type.instance_type)
JitLayerListType = numba.types.List(JitLayer.class_type.instance_type, reflected=True)
