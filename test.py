import pickle
import dense as d
import MNIST_dataloader as mnist
import numpy as np
import cv2

z = 3

loader = mnist.MnistDataloader('CNN\\mnist\\train_images.idx3-ubyte', 'CNN\\mnist\\train_labels.idx1-ubyte',
                                'CNN\\mnist\\test_images.idx3-ubyte', 'CNN\\mnist\\test_labels.idx1-ubyte')

(x_train, y_train), (x_test, y_test) = loader.load_data()
(x_train, y_train), (x_test, y_test) = (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

x_train = np.reshape(x_train/255, (-1, 32, 28, 28))
x_test = np.reshape(x_test/255, (-1, 1, 28, 28))

y_train = np.reshape(y_train, (-1, 32))
y_test = np.reshape(y_test, (-1, 1))

del x_train, y_train

with open("model_mkIII.pickle", "rb") as handle:
    model = pickle.load(handle)

x = np.reshape(x_test, (-1, 28*28))
y = model.forward(x)

check = 0
for i in range(y.shape[0]):
    #print(y[i])
    if y[i].argmax() == y_test[i]:
        check += 1

print(f'val_accuracy: {check}/{y.shape[0]}')
#print(y_test[z])

y_target = np.zeros((y.shape[0], 10))
for i in range(y.shape[0]):
    y_target[i][y_test[i][0]] = 1

loss, d_y = d.MSE(y, y_target)
del d_y

print(loss)

#for i in range(10):
#    cv2.imshow('img', x_test[z][i])
#    cv2.waitKey(0)