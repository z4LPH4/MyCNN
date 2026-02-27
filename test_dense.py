import dense as d
import numpy as np

layer = d.Dense(3*3, 4*4, activation_fn="softmax")
train = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 9, 8, 7, 6, 5, 4, 3, 2]
])/255

print("Input-----------------------------------")
print(train)
print("Output-----------------------------------")
print(layer(train))