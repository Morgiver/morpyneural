
from Neural.JitLayerClass import JitLayer
from Neural.JitNeuralNetworkClass import JitNeuralNetwork

if __name__ == '__main__':
    layer = JitLayer(2, 4, "sigmoid")
    nn = JitNeuralNetwork()
    nn.add_layer(2, 4, "sigmoid")
    nn.add_layer(4, 2, "sigmoid")

    nn2 = JitNeuralNetwork()
    nn2.add_layer(2, 4, "sigmoid")
    nn2.add_layer(4, 2, "sigmoid")

    print("Initial State :")
    print(nn.layers[0].weights.values)
    print(nn2.layers[0].weights.values)
    print(" ")
    print("Evolving...")
    nn.evolve(nn, nn2, 1)
    print(" ")
    print("Evolved State :")
    print(nn.layers[0].weights.values)
    print(nn2.layers[0].weights.values)
