import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import MNIST_dataloader as mnist

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Primo layer dense
        self.fc2 = nn.Linear(128, 64)      # Secondo layer dense
        self.fc3 = nn.Linear(64, 10)       # Ultimo layer dense

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax sull'output finale

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Primo layer con ReLU
        x = self.relu(self.fc2(x))  # Secondo layer con ReLU
        x = self.softmax(self.fc3(x))  # Softmax sul layer finale
        return x

model = SimpleNN()

loader = mnist.MnistDataloader('CNN\\mnist\\train_images.idx3-ubyte', 'CNN\\mnist\\train_labels.idx1-ubyte',
                                'CNN\\mnist\\test_images.idx3-ubyte', 'CNN\\mnist\\test_labels.idx1-ubyte')

(x_train, y_train), (x_test, y_test) = loader.load_data()
(x_train, y_train), (x_test, y_test) = (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

x_train = np.reshape(x_train/255, (-1, 32, 28, 28))
x_test = np.reshape(x_test/255, (-1, 10, 28, 28))

y_train = np.reshape(y_train, (-1, 32))
y_test = np.reshape(y_test, (-1, 10))

# 3. Definizione della loss e dell'ottimizzatore
criterion = nn.MSELoss()  # Funzione di perdita
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Ottimizzatore

# 4. Training del modello
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for images, labels in zip(x_train, y_train):
        # One-hot encoding delle etichette per MSE
        labels_one_hot = np.zeros((32, 10))
        for i in range(labels.size):
            labels_one_hot[i][labels[i]] = 1
        
        images = torch.tensor(images)
        labels_one_hot = torch.tensor(labels_one_hot)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels_one_hot)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch} completata')