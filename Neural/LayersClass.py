from Math.PyMatrix import Py2DMatrix

class Layer:
    def __init__(self):
        """
        Layer contain Weights, Biases and Activation data's of a Neural Network
        """
        self.weights = None
        self.biases = None
        self.activation = None

    def build(self, inputs, nodes, activation):
        """
        Building Weights and Biases matrices, define the Activation Function
        :param inputs:
        :param nodes:
        :param activation:
        :return:
        """
        # Weights PyD2Matrix
        self.weights = Py2DMatrix(nodes, inputs)
        self.weights.build()
        self.weights.randomize(-1, 1)

        # Biases PyD2Matrix
        self.biases = Py2DMatrix(nodes, 1)
        self.biases.build()
        self.biases.randomize(-1, 1)

        # Set Activation
        self.activation = activation

        return self
