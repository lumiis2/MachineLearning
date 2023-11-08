import numpy as np
import matplotlib.pyplot as plt
import torch

x_train = np.array([[1.2], [2.5], [3.1], [3.9], [4.5],
                   [5.0], [5.7], [6.1], [6.8], [7.2],
                   [8.0], [8.5], [9.0], [9.5], [10.1],
                   [10.8], [11.2], [11.9], [12.5], [13.0]],
                   dtype=np.float32)

y_train = np.array([[2.1], [3.4], [3.8], [4.2], [5.0],
                   [5.5], [6.2], [6.8], [7.0], [7.6],
                   [8.5], [9.0], [9.5], [10.0], [10.6],
                   [11.2], [11.8], [12.1], [12.8], [13.2]],
                   dtype=np.float32)

plt.figure(figsize=(8, 8))
plt.scatter(x_train, y_train, c='blue', s=200, label='Dados com tendência linear')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
# plot the inicial data

X_train = torch.from_numpy(x_train) #transforma em tensor torch
Y_train = torch.from_numpy(y_train) #transforma em tensor torch
print('requires_grad for X_train: ', X_train.requires_grad)
print('requires_grad for Y_train: ', Y_train.requires_grad)
# definicao de hiperparametros
input_size = 1 # primeira camda numero de perceptrons
hidden_size = 1 # camdas ocultas
output_size = 1 # camda output
learning_rate = 0.001

#inicializados aleatoriamnete e colocados para requerer gradiente bom para o backpropagation
w1 = torch.rand(input_size, 
                hidden_size, 
                requires_grad=True)
b1 = torch.rand(hidden_size, 
                output_size, 
                requires_grad=True)

for iter in range(1, 4001):
    y_pred = X_train.mm(w1).clamp(min=0).add(b1)
    loss = (y_pred - Y_train).pow(2).sum() 
    if iter % 100 ==0:
        print(iter, loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w1.grad.zero_()
        b1.grad.zero_()

print ('w1: ', w1)
print ('b1: ', b1)

plt.figure(figsize=(8, 8))
plt.scatter(x_train, y_train, c='green', s=200, label='Dados Originais')
plt.plot(x_train, y_pred.detach().numpy(), color='red', linewidth=2, label='Linha Ajustada')
plt.legend()
plt.show()