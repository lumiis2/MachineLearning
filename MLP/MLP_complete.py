import numpy as np
import random
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

 ### FUNCTIONS
def reLU(z):
    return max(0.0, z)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

def CrossEntropyLoss(self):
    pass

class ANN(object):
    def __init__(self, size):
        self.num_layers = len(size)
        self.size = size
        self.biases = [np.random.randn(y, 1) for y in size [1: ]]
        self.weights = [np.random.randn(y, x) for x, y in zip(size [: -1], size[1: ])]

    def feedfoward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+ b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test= len(test_data)
            n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)
        if test_data:
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
        else:
            print("Epoch {0} complete".format(j))

    def backprop(self, x, y):
        deriv_b = [np.zeros(b.shape) for b in self.biases]
        deriv_w = [np.zeros(w.shape) for w in self.weights]
        #feedfoward
        activation = x.reshape(-1, 1)
        activations = [x.reshape(-1, 1)]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backwardpass
        delta = self.cost_derivative(activations[-1], y)*sigmoid_deriv(zs[-1])
        deriv_b[-1] = delta
        deriv_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_deriv(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            deriv_b[-l] = delta
            deriv_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (deriv_b, deriv_w)

    def update(self, mini_batch, eta):
        deriv_b = [np.zeros(b.shape) for b in self.biases]
        deriv_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_deriv_b, delta_deriv_w = self.backprop(x, y)
            deriv_b = [nb+dnb for nb, dnb in zip(deriv_b, delta_deriv_b)]
            deriv_w = [nw+dnw for nw, dnw in zip(deriv_w, delta_deriv_w)]
        self.weights = [w -(eta/len(mini_batch))*nw for w, nw in zip(self.weights, deriv_w)]
        self.biases = [b -(eta/len(mini_batch))*nb for b, nb in zip(self.biases, deriv_b)]
    
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedfoward(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return(output_activations-y)

""""""


import numpy as np
import random
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Define the main function
def main():
    # Define the architecture of the neural network
    model = ANN([784, 30, 10])

    # Define the hyperparameters
    epochs = 30
    mini_batch_size = 64
    learning_rate = 0.1

    # Load the MNIST dataset from the local folder './data'
    training_data, validation_data, test_data = load_mnist_data(train_size=50000, val_size=10000)

    # Train the model using the SGD method
    model.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=validation_data)

    # Evaluate the model on the test data
    accuracy = model.evaluate(test_data)
    print(f'Acurácia no conjunto de teste: {accuracy} / {len(test_data)}')

# Define the load_mnist_data function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import random

def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))

class ANN(nn.Module):
    def __init__(self, size):
        super(ANN, self).__init__()
        self.num_layers = len(size)
        self.size = size
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(y, 1)) for y in size[1:]])
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(y, x)) for x, y in zip(size[:-1], size[1:])])

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(torch.matmul(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)
                
        if test_data:
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=mini_batch_size)
            test_accuracy = self.evaluate(test_loader)
            print(f'Epoch {j}: Test Accuracy: {test_accuracy} / {n_test}')
        else:
            print(f'Epoch {j} complete')

    def backprop(self, x, y):
        deriv_b = [torch.zeros(b.size()) for b in self.biases]
        deriv_w = [torch.zeros(w.size()) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = torch.matmul(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_deriv(zs[-1])
        deriv_b[-1] = delta
        deriv_w[-1] = torch.matmul(delta, activations[-2].t())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_deriv(z)
            delta = torch.matmul(self.weights[-l + 1].t(), delta) * sp
            deriv_b[-l] = delta
            deriv_w[-l] = torch.matmul(delta.view(-1, 1), activations[-l - 1].view(1, -1))
        return (deriv_b, deriv_w)

    def update(self, mini_batch, eta):
        deriv_b = [torch.zeros(b.size()) for b in self.biases]
        deriv_w = [torch.zeros(w.size()) for w in self.weights]
        for x, y in mini_batch:
            x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
            delta_deriv_b, delta_deriv_w = self.backprop(x, y)
            deriv_b = [nb + dnb for nb, dnb in zip(deriv_b, delta_deriv_b)]
            deriv_w = [nw + dnw for nw, dnw in zip(deriv_w, delta_deriv_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, deriv_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, deriv_b)]

    def evaluate(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.feedforward(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def cost_derivative(self, output_activations, y):
        return output_activations - y
    
def load_mnist_data(train_size, val_size):
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=mnist_transform)
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=mnist_transform)

    X_train = [item[0].numpy().flatten() for item in train_dataset]
    y_train = [item[1] for item in train_dataset]
    X_test = [item[0].numpy().flatten() for item in test_dataset]
    y_test = [item[1] for item in test_dataset]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=(val_size + len(X_test)), random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    training_data = list(zip(X_train, y_train))
    validation_data = list(zip(X_val, y_val))
    test_data = list(zip(X_test, y_test))

    return training_data, validation_data, test_data

def main():
    model = ANN([784, 30, 10])
    epochs = 30
    mini_batch_size = 64
    learning_rate = 0.1
    training_data, validation_data, test_data = load_mnist_data(train_size=50000, val_size=10000)
    model.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=validation_data)
    accuracy = model.evaluate(test_data)
    print(f'Accuracy on the test set: {accuracy} / {len(test_data)}')

if __name__ == '__main__':
    main()


