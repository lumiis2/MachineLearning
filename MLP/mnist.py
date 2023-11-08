''' 
Importação de bibliotecas importantes
'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary

''' 
Definição da nossa rede de Multilayer Perceptron (MLP). Nossa rede herda de nn.Module e é construída com
especificações para o tamanho das camadas de entrada, camada oculta e camada de saída.
Para o dataset MNIST, o tamanho da camada de saída é 10, pois estamos classificando dígitos de 0 a 9.
A classe faz uma estrutura genérica em sua inicialização e define as funções que serão utilizadas.
Depois, definimos que para cada camada faremos o uso de nn.Linear, que representa a aplicação da fórmula y = W^T.x + b.
Após isso, definimos como será feito o nosso forward pass: receberemos uma matriz de valores de entrada e faremos um flatten
nela (empilhando suas linhas para formar um vetor), o que gerará nosso vetor de entrada. Em seguida, para cada camada, aplicaremos uma função
de ativação chamada ReLU.
'''
class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
   
        # Função de ativação ReLU e camadas lineares (fully connected)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Achatamos os dados de entrada e aplicamos a função ReLU em camadas sucessivas
        flattened_x = self.flatten(x)
        out = self.relu(self.input_layer(flattened_x))
        out = self.relu(self.hidden_layer(out))
        return self.output_layer(out)

''' 
Aqui instanciamos nosso modelo NeuralNetwork, depois definimos nossa função de custo (CrossEntropyLoss)
e o otimizador (o que percorrerá nosso gradiente).
'''
model = NeuralNetwork(input_size=28*28, hidden_size=64, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

''' 
Aqui carregamos os datasets e os separamos em treino e teste (o proprio pytorch faz essa divisão proporcional) e os transformamos em tensores. 
Depois os preparamos com o DataLoader para podermos utilizá-los em nossa rede, para isso, os dividimos em minibatches
e fazemos um shuffle em seus dados.
'''
train_dataset = datasets.MNIST('./', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./', train=False, download=True, transform=transforms.ToTensor())

train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

''' 
Aqui definimos o número de épocas (quantas vezes o modelo passará sobre o conjunto de treinamento) e definimos o total de minibatches.
'''
num_epochs = 10
total_batch = len(train_dataloader)

# Impressão do resumo da estrutura da rede.
print('Model:')
summary(model, (784,))

''' 
Aqui iteramos por cada época, pegamos os dados de treino em que X seriam as entradas desses dados e y os rótulos relacionados a cada entrada.
Zeramos os gradientes (já que o PyTorch acumula os gradientes e, a cada minibatch, gostaríamos de recalculá-los). Os gradientes ficam armazenados nas camadas do modelo.
Calculamos os outputs chamando nosso modelo, que faz o forward pass resultando nas saídas. Depois calculamos a função de perda e, a partir dela,
fazemos o backward pass, calculando os gradientes e ajustando os pesos e bias chamando o optimizer e fazendo os updates (steps).
Isso tudo acontece a cada minibatch.
'''
model.train()
for epoch in range(num_epochs):
    batch_count = 0

    for X, y in train_dataloader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        batch_count += 1

        # A cada 300 minibatches, a perda é impressa para monitorar o progresso.
        if batch_count % 300 == 0:
            print(f'epoch {epoch}, step: {batch_count}/{total_batch}: loss: {loss.item()}')

''' 
Aqui fazemos o teste da nossa rede depois de treinada.
'''
model.eval()
with torch.no_grad():
    train_acc = []
    test_acc = []

    for X, y in train_dataloader:
        predictions = model(X)
        pred_values, pred_indexes = torch.max(predictions, dim=-1)

        # Cálculo da acurácia comparando as previsões com os rótulos reais.
        acc = sum(pred_indexes == y) / len(y)
        train_acc.append(acc)

    for X, y in test_dataloader:
        predictions = model(X)
        pred_values, pred_indexes = torch.max(predictions, dim=-1)

        # Cálculo da acurácia comparando as previsões com os rótulos reais.
        acc = sum(pred_indexes == y) / len(y)
        test_acc.append(acc)

    print(f'Mean train accuracy: {np.mean(train_acc):.5f}')
    print(f'Mean test accuracy: {np.mean(test_acc):.5f}')