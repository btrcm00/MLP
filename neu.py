import numpy as np
import pandas as pd
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1-x)

class NeuralNetwork:
    def __init__(self, layer_shape = [], lr=0.1):
        if not layer_shape:
            raise ValueError("layer_shape expected non-empty list")
        elif len(layer_shape) < 3:
            raise ValueError("layer_shape expected at least 3 layers (input layer | hidden layers | output layer)")
        if type(lr) is not float:
            raise TypeError("lr expected float type, not %s" % (type(lr)))

        self.layers = layer_shape
        self.lr = lr

        self.W = []
        self.b = []
        for i in range(0, len(layer_shape)-1):
            Wi = np.random.randn(layer_shape[i], layer_shape[i+1])
            b_ = np.zeros((layer_shape[i+1], 1))
            self.W.append(Wi)
            self.b.append(b_)

    def fit_partial(self, X, y):
        num_of_samples = X.shape[0]
        A = [X] 
        #feedfoward
        out = A[-1]
        for i in range(len(self.layers)-1):
            out = sigmoid(np.dot(out, self.W[i]) + self.b[i].T)
            A.append(out)

        #backpropagation
        dJ = [-(y/A[-1] - (1-y)/(1-A[-1]))]#cost function of logistic regression
        dW = []
        db = []
        for i in reversed(range(len(self.layers)-1)):
            dw_ = np.dot(A[i].T, dJ[-1] * d_sigmoid(A[i+1]))
            db_ = np.sum(dJ[-1] * d_sigmoid(A[i+1]), axis=0).reshape(-1,1)
            dj_ = np.dot(dJ[-1] * d_sigmoid(A[i+1]), self.W[i].T)
            dJ.append(dj_)
            dW.append(dw_)
            db.append(db_)

        dW = dW[::-1]
        db = db[::-1]

        for i in range(len(self.layers)-1):
            self.W[i] -= self.lr * dW[i]
            self.b[i] -= self.lr * db[i]

    def fit(self, X, y, epochs, batch_size):
        num_of_samples = X.shape[0]
        if batch_size > num_of_samples:
            batch_size = num_of_samples

        for epoch in range(epochs):
            iterations = np.ceil(num_of_samples / batch_size).astype(int)
            print("Epoch {}/{}:".format(epoch+1, epochs))
            starttime = time.time()

            for iters in range(iterations):
                start_sample_idx = iters * batch_size
                end_sample_idx = (iters+1)*batch_size
                end_sample_idx = end_sample_idx if end_sample_idx <= num_of_samples else num_of_samples 

                self.fit_partial(X[start_sample_idx:end_sample_idx], y[start_sample_idx:end_sample_idx])

            exetime = time.time() - starttime
            loss = self.calculate_loss(X, y)
            print("{}s {}ms/step - loss {}".format(exetime, exetime*1000/iterations, loss))

    def predict(self, X):
        for i in range(len(self.layers)-1):
            X = sigmoid(np.dot(X, self.W[i]) + self.b[i].T)
        return X

    def calculate_loss(self, X, y):
        ypred = self.predict(X)
        return -np.sum(y*np.log(ypred) + (1-y)*np.log(1-ypred))


if __name__ == "__main__":
    data = pd.read_csv('dataset.csv').values
    N, d = data.shape
    X = data[:, 0:d-1]
    y = data[:, d-1].reshape(-1,1)

    p = NeuralNetwork([X.shape[1], 2, 1], 0.1)
    p.fit(X, y, 5000, 100)
    print(np.round(p.predict(X)))