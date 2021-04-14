from Neural.LayersClass import Layer


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def build(self, layers_config):
        """
        Building new Layer Array
        :param layers_config:
        :return:
        """
        for config in layers_config:
            new_layer = Layer().build(config['inputs'], config['nodes'], config['activation'])
            self.layers.append(new_layer)

        return self

    def feed_forward(self, inputs):
        """
        Feed forward inputs into the layers and return the predictions
        :param inputs:
        :return:
        """
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)

        return inputs
