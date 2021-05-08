import numpy as np
import time
import datetime
import random
from morpyneural.Genetic.JitElementClass import JitElement
from morpyneural.Genetic.JitPopulationClass import JitPopulation
from morpyneural.Math.JitMatrix import JitMatrix


if __name__ == '__main__':
    layers_config = np.array([
        [500, 250, 1],
        [250, 250, 1],
        [250, 250, 1],
        [250, 2, 1]
    ], dtype=np.int32)

    element = JitElement()
    element.build(layers_config)

    data = []

    for i in range(500):
        data.append(random.random())

    data = np.array(data, dtype=np.float32)
    data = JitMatrix.from_array(data)

    pop = JitPopulation()
    pop.build(layers_config, 250)
    # pre compiling
    pop.feed_forward(data)
    # Start calculating how much time it take
    print(f"Start at : {datetime.datetime.fromtimestamp(time.time())}")
    print(pop.feed_forward(data))
    print(f"End at : {datetime.datetime.fromtimestamp(time.time())}")
