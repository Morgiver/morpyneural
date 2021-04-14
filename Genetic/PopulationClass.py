from Genetic.ElementClass import Element


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
