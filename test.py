import numpy as np
import numba
import time
import datetime
from Genetic.JitElementClass import JitElement
from Genetic.JitPopulationClass import JitPopulation
from Math.JitMatrix import JitMatrix


@numba.njit
def main():
    layers_config = np.array([
        [2, 4, 1],
        [4, 2, 1]
    ], dtype=np.int32)

    element = JitElement()
    element.build(layers_config)

    print(element.neural_network.layers[0].weights.values)
    data = np.array([2.0, 5.0], dtype=np.float32)
    data = JitMatrix.from_array(data)
    print(element.feed_forward(data).values)

    pop = JitPopulation()
    pop.build(layers_config, 10)
    print(pop.elements)
    print(pop.feed_forward(data))


if __name__ == '__main__':
    layers_config = np.array([
        [50, 250, 1],
        [250, 250, 1],
        [250, 250, 1],
        [250, 250, 1],
        [250, 2, 1]
    ], dtype=np.int32)

    element = JitElement()
    element.build(layers_config)

    data = np.array([2.0, 5.0], dtype=np.float32)
    data = JitMatrix.from_array(data)

    pop = JitPopulation()
    pop.build(layers_config, 250)
    print(pop.feed_forward(data))
    print(f"Start at : {datetime.datetime.fromtimestamp(time.time())}")
    print(pop.feed_forward(data))
    print(f"End at : {datetime.datetime.fromtimestamp(time.time())}")
