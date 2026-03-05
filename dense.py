import numpy as np

class Dense:
    def __init__(self, n, m, **kwargs):
        self.n, self.m = n, m # dimensione in input e output
        self.activation_fn = kwargs.get('activation_fn', None) # ReLU, Softmax...

        # generazione del bias inizializzato a 0
        self.bias = np.zeros((1, m))
        # matrice dei pesi, inizializzata per raggiungere la dimensione di output
        self.weights = np.random.randn(self.n, self.m) * np.sqrt(2/self.n)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x): # x = input/immagine, y = output
        self.x = x # salvo in memoria l'input (per backward)
        # generazione dell'output usando la funzione y = b + W*x
        y = self.bias + x @ self.weights # varianza He per ReLU
        # funzione di attivazione per dare non linearità
        # più avanti si può mettere un if con tutte le funzioni di attivazione
        if self.activation_fn == 'relu':
            self.y = np.maximum(y, 0)
        elif self.activation_fn == 'softmax' or self.activation_fn == 'softmax-crossentropy':
            z = y - np.max(y, axis=1, keepdims=True) # evito la cancellazione numerica sottraendo max
            exps = np.exp(z)
            self.y = exps / np.sum(exps, axis=1, keepdims=True) # il valore massimo
        elif self.activation_fn == None:
            self.y = y
        else:
            raise ValueError('Activation function passata non valida')
        
        return self.y
    
    def backward(self, d_y): # derivata rispetto all'output (si usa chain rule)
        if self.activation_fn == 'relu':
            d_y = d_y * (self.y > 0) # dy/dx=1 se y>0, dW/dx=dW/dy*dy/dx
        elif self.activation_fn == 'softmax': # per la softmax c'è da calcolare la jacobiana
            batch_size = d_y.shape[0] # visto che la derivata non è semplicemente y/x
                                      # bisogna calcolarla per ogni immagine della batch
            for i in range(batch_size):
                prob = self.y[i].reshape(-1, 1) # tolgo una dimensione superflua
                d_softmax = np.diagflat(prob) - np.dot(prob, prob.T) # jacobiana della softmax
                d_y[i] = d_y[i] @ d_softmax
        elif self.activation_fn == None or self.activation_fn == 'softmax-crossentropy':
            pass
        else:
            raise ValueError('Activation function passata non valida')
        # derivata rispetto all'input
        d_x = d_y @ self.weights.T
        # derivata rispetto ai weights
        d_weights = self.x.T @ d_y
        # derivata rispetto al bias
        d_bias = d_y.sum(axis=0)

        return d_x, d_weights, d_bias
    
    def clear_grad(self):
        self.x = None


def MSE(y, y_target):
    loss = np.mean((y - y_target) ** 2) # varianza
    d_y = 2*(y - y_target)/y.shape[1] # derivata rispetto a x (MSE=1/n*sum(x^2))
    return loss, d_y

def cross_entropy(y, y_target):
    y = np.clip(y, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_target*np.log(y))
    loss_v = -y_target*np.log(y)/y.shape[0]
    return loss, loss_v

def SGD(d, param, lr):
    # il gradiente da sempre la massima pendenza
    return (param - lr * d) # convergenza a valori ideali mediante loss

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