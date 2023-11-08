
def get_shapes(arr):
    if isinstance(arr, list):
        return [get_shapes(subarr) for subarr in arr]
    elif isinstance(arr, np.ndarray):
        return arr.shape
    else:
        return None

import random
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[1:], sizes[:-1])]
        
        for i in range(2):
            print(f'Bias da camada {i+1}: {self.biases[i].shape}')
            print(f'Pesos da camada {i+1}: {self.weights[i].shape}')

    def feedforward(self, a):
        a = a.reshape(-1, 1)
        for b, w in zip(self.biases, self.weights):
            #print("b ", get_shapes(b))
            #print("w ", get_shapes(w))
            #print("a ", get_shapes(a))
            a = sigmoid(np.dot(w.transpose(), a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #print("nabla_b ", get_shapes(nabla_b))
        #print("nabla_w ", get_shapes(nabla_w))
        # feedforward
        activation = x.reshape(-1, 1)
        #print("activation ", activation.shape)
        activations = [x.reshape(-1, 1)] # list to store all the activations, layer by layer
        #print("activations ", get_shapes(activations))
        zs = [] # list to store all the z vectors, layer by layer
        #print("zs ", get_shapes(zs))
        for b, w in zip(self.biases, self.weights):
            #print("--w ", get_shapes(w))
            #print("--activation ", get_shapes(activation))
            z = np.dot(w.transpose(), activation)+b
            #print("--z ", get_shapes(z))
            zs.append(z)
            #print("zs ", get_shapes(zs))
            activation = sigmoid(z)
            #print("activation ", get_shapes(activation))
            activations.append(activation)
            #print("activations ", get_shapes(activations))
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        #print("delta ", get_shapes(delta))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()).transpose()
        #print("nabla_b ", get_shapes(nabla_b))
        #print("nabla_w ", get_shapes(nabla_w))
        for l in range(2, self.num_layers):
            z = zs[-l]
            #print(l, "z ", get_shapes(z))
            sp = sigmoid_prime(z)
            #print(l, " sp", get_shapes(sp))
            delta = np.dot(self.weights[-l+1], delta) * sp
            #print(l, "delta ", get_shapes(delta))
            nabla_b[-l] = delta
            #print(l, "nabla_b ", get_shapes(nabla_b))
            #print(l, "nabla_w1 ", get_shapes(nabla_w))
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()).transpose()
            #print(l, "nabla_w2 ", get_shapes(nabla_w))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        print([self.feedforward(x) for (x, y) in test_data])
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        print(test_results)
        return sum(int(x == y) for (x, y) in test_results)
    

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def load_mnist_data(train_size, val_size):
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST('./', train=True, download=False, transform=mnist_transform)
    test_dataset = datasets.MNIST('./', train=False, download=False, transform=mnist_transform)

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
    model = Network([784, 30, 10])
    epochs = 30
    mini_batch_size = 10
    learning_rate = 5.0
    training_data, validation_data, test_data = load_mnist_data(train_size=50000, val_size=10000)
    model.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=validation_data)
    accuracy = model.evaluate(test_data)
    print(f'Accuracy on the test set: {accuracy} / {len(test_data)}')

if __name__ == '__main__':
    main()