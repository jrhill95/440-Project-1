#Jason Hilliard and Jeff Wu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class ThreeLayerNN():
    def __init__(self, num_input, num_hidden, num_output,rlambda):
        self.input_dim = num_input
        self.output_dim = num_output
        self.hidden_dim = num_hidden

        self.rlambda = rlambda
        self.theta1 = np.random.randn(self.input_dim, self.hidden_dim) / np.sqrt(self.input_dim)
        self.bias1 = np.zeros((1, self.hidden_dim))
        self.theta2 = np.random.randn(self.hidden_dim, int(self.output_dim)) / np.sqrt(self.hidden_dim)
        self.bias2 = np.zeros((1, int(self.output_dim)))

    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(0, num_epochs):
            # Forward Propagation
            a1 = X  # input
            z2 = np.dot(a1, self.theta1) + self.bias1  # input to hidden layer
            a2 = np.tanh(z2)  # output from hidden layer
            z3 = np.dot(a2, self.theta2) + self.bias2  # input to output layer

            exp_z = np.exp(z3)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

            # backpropagation
            d3 = np.zeros_like(softmax_scores)
            one_hot_y = np.zeros_like(softmax_scores)
            for i in range(X.shape[0]):
                one_hot_y[i, int(y[i])] = 1

                # error calculation
            d3 = softmax_scores - one_hot_y

            d2 = np.dot(d3, self.theta2.T) * (1 - np.power(a2, 2))

            # gradient computation
            ddtheta2 = np.dot(a2.T, d3)
            ddtheta1 = np.dot(a1.T, d2)

            # parameter updates
            self.theta1 -= learning_rate*ddtheta1 + self.rlambda*self.theta1
            self.theta2 -= learning_rate*ddtheta2 + self.rlambda*self.theta2
            self.bias1 -= learning_rate * np.sum(d2, axis=0)
            self.bias2 -= learning_rate * np.sum(d3, axis=0)

    def compute_cost(self, X, y):
        num_examples = np.shape(X)[0]
        a1 = X  # input
        z2 = np.dot(a1, self.theta1) + self.bias1  # input to hidden layer
        a2 = np.tanh(z2)  # output from hidden layer
        z3 = np.dot(a2, self.theta2) + self.bias2  # input to output layer

        exp_z = np.exp(z3)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        one_hot_y = np.zeros((int(num_examples), int(np.max(y) + 1)))
        logloss = np.zeros((num_examples,))
        for i in range(np.shape(X)[0]):
            one_hot_y[i, int(y[i])] = 1
            logloss[i] = -np.sum(np.log(softmax_scores[i, :]) * one_hot_y[i, :])
        data_loss = np.sum(logloss)
        data_loss += (self.rlambda/2)*(np.sum(np.square(self.theta1))+np.sum(np.square(self.theta2)))
        return 1. / num_examples * data_loss

    def predict(self, X):
        a1 = X  # input
        z2 = np.dot(a1, self.theta1) + self.bias1  # input to hidden layer
        a2 = np.tanh(z2)  # output from hidden layer
        z3 = np.dot(a2, self.theta2) + self.bias2  # input to output layer

        exp_z = np.exp(z3)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis=1)

        return predictions