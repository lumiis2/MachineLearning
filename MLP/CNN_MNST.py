import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from MLPeCNN_from_scratch import *
import matplotlib.pyplot as plt

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train_and_plot(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    loss_history = []

    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        loss_history.append(error)

        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

    # Plotar a curva de aprendizado
    plt.plot(loss_history)
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return loss_history

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    convolutional_layer((1, 28, 28), 3, 5),
    Sigmoid(),
    reshape_layer((5, 26, 26), (5 * 26 * 26, 1)),
    dense_layer(5 * 26 * 26, 100),
    Sigmoid(),
    dense_layer(100, 2),
    Sigmoid()
]

# train
train_and_plot(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

correct_predictions = 0
total_samples = len(x_test)

for x, y in zip(x_test, y_test):
    output = predict(network, x)
    predicted_label = np.argmax(output)
    true_label = np.argmax(y)

    if predicted_label == true_label:
        correct_predictions += 1

accuracy = correct_predictions / total_samples
print(f"Acurácia: {accuracy}")


