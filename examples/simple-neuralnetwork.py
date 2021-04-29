import numpy as np
import time
import datetime
from morpyneural.Genetic.JitElementClass import JitElement
from morpyneural.Genetic.JitPopulationClass import JitPopulation
from morpyneural.Math.JitMatrix import JitMatrix


def main():
    layers_config = np.array([
        [2, 250, 1],
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


if __name__ == '__main__':
    layers_config = np.array([
        [2, 250, 1],
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
