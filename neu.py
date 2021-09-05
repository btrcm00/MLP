import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_openml

class NeuralNetwork:
    """
    activation: Activation function for the hidden layer.
    lr: Learning rate schedule for weight updates.
    batch_size: Size of minibatches for stochastic optimizers
    alpha_activation: Only used when activation='prelu' or 'elu'
    alpha: L2 penalty (regularization term) parameter.
    """
    def __init__(self, layer_shape = [], lr=0.1, activation='logistic', alpha_activation=0.01, batch_size='auto', alpha = 0.0001):
        if not layer_shape:
            raise ValueError("layer_shape expected non-empty list")
        elif len(layer_shape) < 3:
            raise ValueError("layer_shape expected at least 3 layers (input layer | hidden layers | output layer)")
        if type(lr) is not float:
            raise TypeError("lr expected float type, not %s" % (type(lr)))

        if activation not in ['relu', 'logistic', 'tanh', 'prelu', 'elu']:
            self.activation = 'logistic'
        else:
            self.activation = activation

        self.batch_size = batch_size
        self.alpha = alpha 

        #avtivation function for outputlayer
        if layer_shape[-1] in [1, 2]:
            self.out_activation = 'sigmoid'
        else:
            self.out_activation = 'softmax'

        self.layers = layer_shape
        self.learning_rate = lr
        self.alpha_activation = alpha_activation

        #random initialization weights and bias
        self.__W = []
        self.__b = []
        for i in range(0, len(layer_shape)-1):
            Wi = np.random.randn(layer_shape[i], layer_shape[i+1])
            bi = np.zeros((layer_shape[i+1], 1))
            self.__W.append(Wi)
            self.__b.append(bi)

    def __hidden_activation(self, x):
        if self.activation == 'logistic':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        elif self.activation == 'prelu':
            x[x[:]<0] *= self.alpha_activation
            return x
        elif self.activation == 'elu':
            x[x[:]<0] = self.alpha_activation * (np.exp(x[x[:]<0]) - 1)
            return x

    def __hidden_activation_derivative(self, x):
        if self.activation == 'logistic':
            return x * (1 - x)
        elif self.activation == 'relu':
            x[x[:]>=0]=1
            x[x[:]<0]=0
            return x
        elif self.activation == 'tanh':
            return 1 - x**2
        elif self.activation == 'prelu':
            x[x[:]<0] = self.alpha_activation
            x[x[:]>=0]=1
            return x
        elif self.activation == 'elu':
            x[x[:]>0] = 1
            x[x[:]<0] += self.alpha_activation
            return np.maximum(0.01 * x, x)

    def __output_activation(self, x):
        if self.out_activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.out_activation == 'softmax':
            return np.exp(x) / np.sum(np.exp(x))

    def __output_activation_derivative(self,x):
        if self.out_activation == 'sigmoid':
            return x * (1-x)
        elif self.out_activation == 'softmax':
            return np.diagflat(x) - np.dot(x, x.T)

    def __fit_partial(self, X, y):
        num_of_samples = X.shape[0]
        A = [X]
        #feedfoward
        a_i = A[-1]
        for i in range(len(self.layers)-1):
            #sigmoid  function for output layer
            if i == len(self.layers)-2:
                a_i = self.__output_activation(np.dot(a_i, self.__W[i]) + self.__b[i].T)
            #activation function for hidden layers
            else:
                a_i = self.__hidden_activation(np.dot(a_i, self.__W[i]) + self.__b[i].T)
            A.append(a_i)

        #backpropagation
        epsilon = 0.00001 #a number to advoid division by zero
        dJ = [-(y/(A[-1] + epsilon) - (1-y)/(1-A[-1] + epsilon))]
        dW = []
        db = []
        for i in reversed(range(len(self.layers)-1)):
            if i == len(self.layers)-2:
                dw_ = np.dot(A[i].T, dJ[-1] * self.__output_activation_derivative(A[i+1]))
                db_ = np.sum(dJ[-1] * self.__output_activation_derivative(A[i+1]), axis=0).reshape(-1,1)
                dj_ = np.dot(dJ[-1] * self.__output_activation_derivative(A[i+1]), self.__W[i].T)
            else:
                dw_ = np.dot(A[i].T, dJ[-1] * self.__hidden_activation_derivative(A[i+1]))
                db_ = np.sum(dJ[-1] * self.__hidden_activation_derivative(A[i+1]), axis=0).reshape(-1,1)
                dj_ = np.dot(dJ[-1] * self.__hidden_activation_derivative(A[i+1]), self.__W[i].T)
            dJ.append(dj_)
            dW.append(dw_)
            db.append(db_)

        dW = dW[::-1]
        db = db[::-1]

        for i in range(len(self.layers)-1):
            self.__W[i] = self.__W[i]*(1 - self.learning_rate * self.alpha / num_of_samples) - self.learning_rate * dW[i]
            self.__b[i] -= self.learning_rate * db[i]

    def fit(self, X, y, epochs, batch_size):
        num_of_samples = X.shape[0]
        if self.batch_size == 'auto':
            batch_size = min(200, num_of_samples)
        if batch_size > num_of_samples:
            batch_size = num_of_samples

        for epoch in range(epochs):
            iterations = np.ceil(num_of_samples / batch_size).astype(int)
            print("Epoch {}/{}:".format(epoch+1, epochs))
            start_epoch_time = time.time()

            for iters in range(iterations):
                start_sample_idx = iters * batch_size
                end_sample_idx = (iters+1)*batch_size
                end_sample_idx = end_sample_idx if end_sample_idx <= num_of_samples else num_of_samples 

                self.__fit_partial(X[start_sample_idx:end_sample_idx], y[start_sample_idx:end_sample_idx])

            epoch_training_time = time.time() - start_epoch_time
    
            ypred = self.predict(X)
            loss = self.loss(y, ypred)
            print("{}s {}ms/step - loss {}".format(epoch_training_time, epoch_training_time*1000/iterations, loss))

    def predict(self, X):
        for i in range(len(self.layers)-1):
            if i == len(self.layers)-2:
                X = self.__output_activation(np.dot(X, self.__W[i]) + self.__b[i].T)
            else:
                X = self.__hidden_activation(np.dot(X, self.__W[i]) + self.__b[i].T)
        return X

    def loss(self, y, ypred):
        epsilon = 0.0001#a number to advoid division by zero
        return np.mean(-(y*np.log(ypred+epsilon) + (1-y)*np.log(1-ypred+epsilon)))

    def score(self, X, y):
        if self.layers[-1] > 1:
            ypred = np.argmax(self.predict(X),axis=1).reshape(-1,1)
        else:
            ypred = np.around(self.predict(X))
        sc = y == ypred
        return np.sum(sc) / len(sc)

if __name__ == "__main__":
    data = pd.read_csv('dataset.csv').values
    N, d = data.shape
    X = data[:, 0:d-1]
    y = data[:, d-1].reshape(-1,1)

    score = {}
    for i in ['tanh', 'logistic', 'relu', 'elu', 'prelu']:
        sta = time.time()
        mlp = NeuralNetwork([X.shape[1], 3, 3, 1], 0.01, activation=i)
        mlp.fit(X[:N-3], y[:N-3], 1000, 100)
        score[i] = [mlp.score(X[N-3:], y[N-3:]), mlp.loss(y[N-3:], mlp.predict(X[N-3:])), time.time() - sta]

    for i in score:
        print("%s activation function output score: %s - Loss: %s - Training time: %ss" % (i, score[i][0], score[i][1], score[i][2]))