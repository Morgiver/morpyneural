from Genetic.ElementClass import Element
from Math.PyMatrix import Py2DMatrix


class Population:
    def __init__(self):
        self.elements = []

    def build(self, layers_configuration, max_elements):
        """
        Building new set of Elements
        :param layers_configuration:
        :param max_elements:
        :return:
        """
        self.elements.clear()
        for i in range(max_elements):
            new_element = Element().build(layers_configuration)
            self.elements.append(new_element)

    def feed_forward(self, inputs):
        """
        Feed forwarding all Elements
        :param inputs:
        :return:
        """
        inputs = Py2DMatrix(1, 1).from_array(inputs)

        results = []
        for element in self.elements:
            results.append(element.feed_forward(inputs))

        return results
