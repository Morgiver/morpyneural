from Neural.JitLayerClass import JitLayer, JitLayerType

if __name__ == '__main__':
    layer = JitLayer(2, 4, "sigmoid")
    print(layer.weights.values)
