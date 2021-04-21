from numba.experimental import jitclass
from Neural.JitLayerClass import JitLayer, JitLayerListType


spec = [
    ('layers', JitLayerListType)
]


@jitclass(spec)
class JitNeuralNetwork(object):
    def __init__(self, layers_config):
        self.layers = [JitLayer(1, 1, "sigmoid")]
        self.layers.pop()

        for config in layers_config:
            new_layer = JitLayer(config['inputs'], config['nodes'], config['activation'])
            self.layers.insert(len(self.layers), new_layer)

    def feed_forward(self, inputs):
        """
        Feed forward inputs into the layers and return the predictions
        :param inputs:
        :return:
        """
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)

        return inputs

    def evolve(self, parent_a, parent_b, learning_rate=0.001):
        """
        Evolving all layers by crossovering the two parents and mutate the matrices
        :param parent_a:
        :param parent_b:
        :param learning_rate:
        :return:
        """
        for i in range(len(parent_a.layers)):
            self.layers[i].evolve(parent_a.layers[i], parent_b.layers[i], learning_rate)

        return self
