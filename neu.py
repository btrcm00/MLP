import numpy as np
import pandas as pd
import time

class NeuralNetwork:
    """
    activation: Activation function for the hidden layer.
    lr: Learning rate schedule for weight updates.
    """
    def __init__(self, layer_shape = [], lr=0.1, activation='logistic'):
        if not layer_shape:
            raise ValueError("layer_shape expected non-empty list")
        elif len(layer_shape) < 3:
            raise ValueError("layer_shape expected at least 3 layers (input layer | hidden layers | output layer)")
        if type(lr) is not float:
            raise TypeError("lr expected float type, not %s" % (type(lr)))
        if activation not in ['relu', 'logistic', 'tanh']:
            self.activation = 'logistic'
        else:
            self.activation = activation

        self.layers = layer_shape
        self.lr = lr

        #random initialization weights and bias
        self.__W = []
        self.__b = []
        for i in range(0, len(layer_shape)-1):
            Wi = np.random.randn(layer_shape[i], layer_shape[i+1])
            bi = np.zeros((layer_shape[i+1], 1))
            self.__W.append(Wi)
            self.__b.append(bi)

    def __activation(self, x):
        if self.activation == 'logistic':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __activation_derivative(self, x):
        if self.activation == 'logistic':
            return x * (1 - x)
        elif self.activation == 'relu':
            x[x[:]>0]=1
            x[x[:]<=0]=0
            return x
        elif self.activation == 'tanh':
            return 1 - x**2
            
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def __fit_partial(self, X, y):
        num_of_samples = X.shape[0]
        A = [X] 
        #feedfoward
        a_i = A[-1]
        for i in range(len(self.layers)-1):
            #sigmoid  function for output layer
            if i == len(self.layers)-2:
                a_i = self.__sigmoid(np.dot(a_i, self.__W[i]) + self.__b[i].T)
            #activation function for hidden layers
            else:
                a_i = self.__activation(np.dot(a_i, self.__W[i]) + self.__b[i].T)
            A.append(a_i)

        #backpropagation
        dJ = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dW = []
        db = []
        for i in reversed(range(len(self.layers)-1)):
            if i == len(self.layers)-2:
                dw_ = np.dot(A[i].T, dJ[-1] * self.__sigmoid_derivative(A[i+1]))
                db_ = np.sum(dJ[-1] * self.__sigmoid_derivative(A[i+1]), axis=0).reshape(-1,1)
                dj_ = np.dot(dJ[-1] * self.__sigmoid_derivative(A[i+1]), self.__W[i].T)
            else:
                dw_ = np.dot(A[i].T, dJ[-1] * self.__activation_derivative(A[i+1]))
                db_ = np.sum(dJ[-1] * self.__activation_derivative(A[i+1]), axis=0).reshape(-1,1)
                dj_ = np.dot(dJ[-1] * self.__activation_derivative(A[i+1]), self.__W[i].T)
            dJ.append(dj_)
            dW.append(dw_)
            db.append(db_)

        dW = dW[::-1]
        db = db[::-1]

        for i in range(len(self.layers)-1):
            self.__W[i] -= self.lr * dW[i]
            self.__b[i] -= self.lr * db[i]

    def fit(self, X, y, epochs, batch_size):
        num_of_samples = X.shape[0]
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
                X = self.__sigmoid(np.dot(X, self.__W[i]) + self.__b[i].T)
            else:
                X = self.__activation(np.dot(X, self.__W[i]) + self.__b[i].T)
        return X

    def loss(self, y, ypred):
        return np.mean(-(y*np.log(ypred) + (1-y)*np.log(1-ypred)))

    def score(self, X, y):
        sc = y == np.around(self.predict(X))
        return np.sum(sc) / len(sc)

if __name__ == "__main__":
    data = pd.read_csv('dataset.csv').values
    N, d = data.shape
    X = data[:, 0:d-1]
    y = data[:, d-1].reshape(-1,1)

    score = {}
    for i in ['tanh', 'logistic', 'relu']:
        sta = time.time()
        mlp = NeuralNetwork([X.shape[1], 2, 1], 0.01, activation=i)
        mlp.fit(X, y, 5000, 100)
        score[i] = [mlp.score(X, y), time.time() - sta]

    for i in score:
        print("%s activation function output score: %s - Training time: %ss" % (i, score[i][0], score[i][1]))