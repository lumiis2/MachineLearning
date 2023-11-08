import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets 

def unit_step_func(x):
    return np.where(x>0, 1, 0)

class Perceptron:

    def __init__(self, learning_rate = 0.01, n_iters=1000):
        self.lr =  learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = 2 * np.random.random(size=n_features) - 1
        self.b = 2 * np.random.random() - 1

        y_ = np.where(y > 0, 1, 0)

        #learn w
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.w) + self.b
                y_hat = self.activation_func(linear_output)

                #perceptron update
                update = self.lr * (y_[idx] - y_hat)
                self.w += update * x_i
                self.b += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        y_hat = self.activation_func(linear_output)
        return y_hat


    

# Testing
if __name__ == "__main__":  

    def train_split(X, y, test_size=0.2, random_seed=None):

        if random_seed is not None:
            np.random.seed(random_seed)

        num_samples = len(X) #total de amostras
        num_test_samples = int(test_size * num_samples) #amostras de teste
        test_indices = np.random.choice(num_samples, size=num_test_samples, replace=False)  #Índices aleatórios 
        train_indices = np.setdiff1d(np.arange(num_samples), test_indices) #pego a diferenca dos arrays (faco o complemento 80/20)

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
            n_samples=150, n_features=2, centers=2, cluster_std=1.6, random_state=np.random.randint(100)
    )
    X_train, X_test, y_train, y_test = train_split(X, y, test_size=0.2, random_seed=None)

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    x_classA = X_train[y_train == 0]
    x_classB = X_train[y_train == 1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plotando "x" azuis para a classe 0
    plt.scatter(x_classA[:, 0], x_classA[:, 1], marker="x", c='blue', label="Classe A")

    # Plotando "o" vermelhas para a classe 1
    plt.scatter(x_classB[:, 0], x_classB[:, 1], marker="o", c='red', label="Classe B")

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.w[0] * x0_1 - p.b) / p.w[1]
    x1_2 = (-p.w[0] * x0_2 - p.b) / p.w[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.legend()
    plt.show()
