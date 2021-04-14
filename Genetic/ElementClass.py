from Neural.NeuralNetworkClass import NeuralNetwork


class Element:
    def __init__(self):
        """
        Element is a part of Population, it containing the neural network and (will) handle
        the score logic
        """
        self.active = True
        self.neural_network = NeuralNetwork()

    def build(self, layers_configuration):
        """
        Build the neural network
        :param layers_configuration:
        :return:
        """
        self.neural_network.build(layers_configuration)
