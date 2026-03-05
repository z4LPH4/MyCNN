import numpy as np
from src.layers.dense import Dense
from src.layers.dense import SGD

class SimpleMLP:
    def __init__(self):
        self.layers = []

        self.layers.append(Dense(28*28, 128, activation_fn='relu'))
        self.layers.append(Dense(128, 64, activation_fn='relu'))
        self.layers.append(Dense(64, 10, activation_fn='softmax-crossentropy'))

        self.layers = np.array(self.layers)
        # learning rate
        self.lr = 0.001 # per una rete piccola 10^-3 va bene, ma può essere meno
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x
    
    def backward(self, d_y):
        grad = []

        for layer in reversed(self.layers):
            d_y, d_weights, d_bias = layer.backward(d_y)
            grad.insert(0, [d_weights, d_bias])
        
        return grad

    def update(self, grad):
        for layer, (d_weights, d_bias) in zip(self.layers, grad):
            layer.weights = SGD(d_weights, layer.weights, self.lr)
            layer.bias = SGD(d_bias, layer.bias, self.lr)
            layer.clear_grad()