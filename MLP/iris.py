import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


iris_data = load_iris()

n_samples, n_features = iris_data.data.shape
def Show_Diagram(_x ,_y,title):
    plt.figure(figsize=(10,4))
    plt.scatter(iris_data.data[:,_x], 
    iris_data.data[:, _y], c=iris_data.target, cmap=cm.viridis)
    plt.xlabel(iris_data.feature_names[_x]);  
    plt.ylabel(iris_data.feature_names[_y]); 
    plt.title(title)
    plt.colorbar(ticks=([0, 1, 2]));
    plt.show();
Show_Diagram(0,1,'Sepal')
Show_Diagram(2,3,'Petal')




class ArtificialNeuralNetwork(nn.Module):
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

model = ArtificialNeuralNetwork(input_size=4, hidden_size=50, output_size=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X = iris_data.data
y = iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int64)

num_epochs = 100

# Impressão do resumo da estrutura da rede.
print('Model:')
summary(model, (4,))

model.train()
for epoch in range(num_epochs):
   
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

''' 
Aqui fazemos o teste da nossa rede depois de treinada.
'''
model.eval()
with torch.no_grad():
    train_acc = []
    test_acc = []

    predictions = model(X_train)
    pred_values, pred_indexes = torch.max(predictions, dim=-1)

    # Cálculo da acurácia comparando as previsões com os rótulos reais.
    train_acc = (pred_indexes == y_train).sum().item() / len(y_train)

    predictions = model(X_test)
    pred_values, pred_indexes = torch.max(predictions, dim=-1)

    # Cálculo da acurácia comparando as previsões com os rótulos reais.
    test_acc = (pred_indexes == y_test).sum().item() / len(y_test)

print(f'Mean train accuracy: {train_acc:.5f}')
print(f'Mean test accuracy: {test_acc:.5f}')
