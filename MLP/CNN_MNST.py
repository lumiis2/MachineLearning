import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from keras.utils import to_categorical
from MLPeCNN_from_scratch import * 

def _predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def _train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = _predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

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



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.network = nn.ModuleList([
            convolutional_layer((1, 28, 28), 3, 5),
            Sigmoid(),
            reshape_layer((3, 24, 24), (3 * 24 * 24,)),
            dense_layer(3 * 24 * 24, 100),
            Sigmoid(),
            dense_layer(100, 10),  # Ajuste para o número de classes (10 para MNIST)
            Softmax()
        ])

# Carregamento dos dados usando PyTorch
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./', train=False, download=True, transform=transform)

x_train, x_val, y_train, y_val = train_test_split(train_dataset.data, train_dataset.targets, test_size=0.2, random_state=42)

# Pré-processamento dos dados
x_train, y_train = preprocess_data(x_train.numpy(), y_train.numpy(), 100)
x_val, y_val = preprocess_data(x_val.numpy(), y_val.numpy(), 100)
x_test, y_test = preprocess_data(test_dataset.data.numpy(), test_dataset.targets.numpy(), 100)

# Converter para torch.Tensor
x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
x_val, y_val = torch.from_numpy(x_val), torch.from_numpy(y_val)
x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

# Criar DataLoader para treinamento
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Criar instância da CNN
model = CNN()

# Definir otimizador e função de perda
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Treinamento do modelo
model.train()
for epoch in range(10):  # Número de épocas
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()

# Avaliação no conjunto de validação
model.eval()
with torch.no_grad():
    predictions_val = np.argmax(_predict(model.network, x_val.numpy()), axis=1)
    accuracy_val = accuracy_score(np.argmax(y_val.numpy(), axis=1), predictions_val)
    print(f'Validation Accuracy: {accuracy_val * 100:.2f}%')

# Avaliação no conjunto de teste
predictions_test = np.argmax(_predict(model.network, x_test.numpy()), axis=1)
accuracy_test = accuracy_score(np.argmax(y_test.numpy(), axis=1), predictions_test)
print(f'Test Accuracy: {accuracy_test * 100:.2f}%')