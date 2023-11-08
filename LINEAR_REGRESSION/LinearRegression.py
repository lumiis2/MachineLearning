import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Função para gerar dados sintéticos
def synthetic_data(w, b, num_examples):  
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# Converter os dados numpy em tensores
X_train = features[:, 1].view(-1, 1)
Y_train = labels.view(-1, 1)

# Definir a classe Perceptron
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

# Inicializar o modelo Perceptron
input_size = 1  # Agora, temos apenas 1 feature
model = Perceptron(input_size)

# Definir a função de perda (loss function) e otimizador
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Treinamento do modelo
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)

    # Backward pass e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Avaliação do modelo
model.eval()
with torch.no_grad():
    predicted = model(X_train)

# Plotagem dos resultados
plt.figure(figsize=(8, 8))
plt.scatter(X_train.view(-1).detach().numpy(), Y_train.view(-1).detach().numpy(), s=10, label='Dados com tendência linear')
plt.plot(X_train.view(-1).detach().numpy(), predicted.view(-1).detach().numpy(), color='red', linewidth=2, label='Linha Ajustada')
plt.xlabel('Feature')
plt.ylabel('Labels')
plt.legend()
plt.show()
