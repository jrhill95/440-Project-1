# Jason Hilliard and Jeff Wu
import numpy as np
import matplotlib.pyplot as plt



class TwoLayerNN():
    def __init__(self, input_dim, output_dim):
        self.theta = np.random.randn(input_dim, int(output_dim)) / np.sqrt(input_dim)
        self.bias = np.zeros((1, int(output_dim)))

    # --------------------------------------------------------------------------

    def compute_cost(self, X, y):

        num_examples = np.shape(X)[0]
        z = np.dot(X, self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        one_hot_y = np.zeros((num_examples, int(np.max(y)) + 1))
        logloss = np.zeros((num_examples,))
        for i in range(np.shape(X)[0]):
            one_hot_y[i, int(y[i])] = 1
            logloss[i] = -np.sum(np.log(softmax_scores[i, :]) * one_hot_y[i, :])
        data_loss = np.sum(logloss)
        return 1. / num_examples * data_loss

    # --------------------------------------------------------------------------

    def predict(self, X):

        z = np.dot(X, self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / (exp_z + 1)
        predictions = np.argmax(softmax_scores, axis=1)
        return predictions

    # --------------------------------------------------------------------------

    def train(self, X, y, num_epochs, lr=0.01):
        for epoch in range(0, num_epochs):

            # Forward propagation
            z = np.dot(X, self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

            # Backpropagation
            beta = np.zeros_like(softmax_scores)
            one_hot_y = np.zeros_like(softmax_scores)
            for i in range(X.shape[0]):
                one_hot_y[i, int(y[i])] = 1
            beta = softmax_scores - one_hot_y

            # Compute gradients of model parameters
            dtheta = np.dot(X.T, beta)
            dbias = np.sum(beta, axis=0)

            # Gradient descent parameter update
            self.theta -= lr * dtheta
            self.bias -= lr * dbias
