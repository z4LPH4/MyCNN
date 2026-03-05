import pickle
import src.layers.dense as d
from src.layers.SimpleMLP import SimpleMLP
import time
from tqdm import tqdm
import utils.MNIST_dataloader as mnist
import numpy as np
from sklearn.utils import shuffle
#import torch
#import torch.nn as nn

mlp = SimpleMLP()

loader = mnist.MnistDataloader('mnist\\train_images.idx3-ubyte', 'mnist\\train_labels.idx1-ubyte',
                                'mnist\\test_images.idx3-ubyte', 'mnist\\test_labels.idx1-ubyte')

(x_train, y_train), (x_test, y_test) = loader.load_data()
(x_train, y_train), (x_test, y_test) = (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

x_train = np.reshape(x_train/255, (-1, 32, 28, 28))
x_test = np.reshape(x_test/255, (-1, 10, 28, 28))

y_train = np.reshape(y_train, (-1, 32))
y_test = np.reshape(y_test, (-1, 10))

total_epochs = 20


for epoch in tqdm(range(total_epochs), desc='Training process'):
    start_time = time.time()
    loss = None

    x_train, y_train = shuffle(x_train, y_train)

    for i in range(x_train.shape[0]):
        batch = x_train[i]
        label = y_train[i]

        x = np.reshape(batch, (32, -1))
        pred = mlp.forward(x)
        y_target = np.zeros((32, 10))
        for j in range(label.size):
            y_target[j][label[j]] = 1
        #loss, d_y = d.MSE(pred, y_target)
        loss, d_y = d.cross_entropy(pred, y_target)
        d_y = (pred - y_target) / x.shape[0]

        if i % 5 == 0:
            #grad = mlp.backward(d_y)
            grad = mlp.backward(d_y)
            if i != 0:
                mlp.update(grad)
        else:
            #grad += mlp.backward(d_y)
            grad = mlp.backward(d_y)

    elapsed_time = time.time() - start_time
    remaining_time = (total_epochs - (epoch + 1)) * (elapsed_time/(epoch + 1))
    tqdm.write(f'Epoch {epoch + 1}/{total_epochs} completed, loss: {loss}, ETA: {remaining_time:.2f} s')

with open('models\\model.pickle', 'wb') as handle:
    pickle.dump(mlp, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
fc1 = nn.Linear(4, 2, bias=True)

x = torch.randn(1, 4, requires_grad=True)  # Batch di 32 esempi
y = fc1(x)

target = torch.randn(1, 2)  # Target casuale
loss = nn.MSELoss()(y, target)  # Loss MSE
loss_vector = nn.MSELoss(reduction='none')(y, target)  # Loss MSE
loss.backward()

mio = d.Dense(4, 2)
mio.bias = fc1.bias.detach().numpy()
mio.weights = fc1.weight.detach().numpy().T

y1 = mio(x.detach().numpy())
loss1 = d.MSE(y1, target.detach().numpy())

d_x, d_W, d_b = mio.backward(2*(y1-target.detach().numpy())/y1.shape[1])

#print(fc1.bias)
#print(fc1.weight)

#print(mio.bias)
#print(mio.weights)

print(y)
print(y1)

print(loss_vector)
print(loss1)

#print("Gradiente dei pesi:", fc1.weight.grad)
print("Gradiente del bias:", fc1.bias.grad)

#print("Gradiente MIO dei pesi:", d_W)
print("Gradiente MIO del bias:", d_b)'''