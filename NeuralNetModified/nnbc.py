''' turn all of this into a flask app that i can host on my website? or write it in javascript?
supposedly you can do python & flask, but you "need" a vps or dedicated server, which i dont have...

https://help.dreamhost.com/hc/en-us/articles/216137717-Python-overview
http://flask.pocoo.org/
https://help.dreamhost.com/hc/en-us/articles/216128557-Guidelines-for-setting-up-a-Python-file-at-DreamHost
https://help.dreamhost.com/hc/en-us/articles/217956197-Python-FastCGI
https://mattcarrier.com/flask-dreamhost-setup/
https://discussion.dreamhost.com/t/flask-and-fastcgi/61884/2

get digits working as well
'''

from abc import ABC, abstractmethod
from twolnn import TwoLayerNN as nn2
from threelnn import ThreeLayerNN as nn3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color




def selectEpochs():
    while True:
        try:
            epochs = int(input("Enter number of iterations:\n"))
            if epochs>0:
                return epochs
        except:
            pass
        print('Please enter a positive integer')

def selectLearningRate():
    while True:
        try:
            rate = float(input("Enter learning rate:\n"))
            if rate>0:
                return rate
        except:
            pass
        print('Please enter a positive decimal value')

def selectHiddenDim():
    while True:
        try:
            nodes = int(input("Enter number of hidden nodes:\n"))
            if nodes >=0:
                return nodes
        except:
            pass
        print('Please enter a positive integer')

def selectRLambda():
    return 
    while True:
        try:
            rlambda = float(input("Enter regularization lambda:\n"))
            if rlambda>=0:
                return rlambda
        except:
            pass
        print('Please enter a non-negative decimal value')

def selectDataSet():
    while True:
        data = input("Linear or nonlinear dataset? l/n\n")
        if data=='l':
            datasetX = "DATA/LinearX.csv"
            datasety = "DATA/Lineary.csv"
        if data=='n':
            datasetX = "DATA/NonLinearX.csv"
            datasety = "DATA/NonLineary.csv"
        try:
            X = np.genfromtxt(datasetX, delimiter=',')
            y = np.genfromtxt(datasety, delimiter=',')
            return X,y
        except:
            pass
        print('Please enter an "l" or an "n"')

def splitData(X,y):
    # from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
    return Xtrain,Xtest,ytrain,ytest


def plot_decision_boundary(model, X, y):
    cmap2 = color.ListedColormap(['#002233','#efff2b'])
    cmap3 = color.ListedColormap(['#004d80', '#f4ff77'])
    #f0ff37
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=cmap3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap2, alpha=0.8)
    plt.show()

def confusionAndAccuracy(model, X, y, outputDim):
    acc = 0
    yPred = model.predict(X)
    conMat = np.zeros((int(outputDim), int(outputDim)))
    for i in range(len(yPred)):
        conMat[yPred[i], int(y[i])] += 1
        if y[i] == yPred[i]:
            acc += 1
    acc = acc / len(yPred)
    return acc, conMat





def runExperiment(X,y,numEpochs,learningRate,hiddenDim,rlambda):
    print('running experiment...')
    Xtrain, Xtest, ytrain, ytest = splitData(X,y)
    inputDim = np.shape(X)[1]
    outputDim = np.max(y) + 1
    if hiddenDim==0:
        neuralNet = nn2(inputDim, outputDim)
    else:
        neuralNet = nn3(inputDim,hiddenDim,outputDim,rlambda)

    neuralNet.train(Xtrain,ytrain,numEpochs,learningRate)
    trainAcc, trainConMat = confusionAndAccuracy(neuralNet, Xtrain, ytrain, outputDim)
    trainCost = neuralNet.compute_cost(Xtrain, ytrain)
    testAcc, testConMat = confusionAndAccuracy(neuralNet,Xtest,ytest,outputDim)
    cost = neuralNet.compute_cost(Xtest,ytest)

    print('TRAINING DATA ACCURACY: ', trainAcc)
    print('TRAINING DATA CONFUSION MATRIX: \n', trainConMat)
    print('TRAINING DATA COST: \n', trainCost)
    print('TEST DATA ACCURACY: ', testAcc)
    print('TEST DATA CONFUSION MATRIX: \n', testConMat)
    print('TEST DATA COST: \n',cost)
    plot_decision_boundary(neuralNet,X,y)

def main():
    numEpochs = selectEpochs()
    learningRate = selectLearningRate()
    hiddenDim = selectHiddenDim()
    rlambda = selectRLambda()
    X,y = selectDataSet()

    runExperiment(X, y, numEpochs, learningRate, hiddenDim, rlambda)


if __name__=="__main__":
    main()
