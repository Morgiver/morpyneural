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
