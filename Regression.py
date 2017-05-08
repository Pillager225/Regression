#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
import math

learningRate = 1e-6
kFoldSplit = 5
startingLamb = int(400)

# parses data and splits into table header, data, and labels
def getData(path):
    raw = np.array(pd.read_csv(path, delimiter='\t', header=-1))
    columnNames = raw[0,:]
    X = np.array(raw[1:,:-1], dtype=np.float32)
    y = np.array(raw[1:,len(raw[0])-1], dtype=np.float32)[np.newaxis].T
    return columnNames, X, y

# returs an array that is X with another column vector of 1s
def addBiasDummyFeature(X):
    newX = np.zeros((len(X),len(X[0])+1), dtype=np.float32)
    newX[:,:-1] = X
    newX[0:,len(X[0])] = np.ones(X.shape[0])
    return newX

def L2Norm(V):
    return np.dot(V.T, V)

def getMSE(X, y, w):
    ypredict = np.dot(X, w)
    ydiff = y-ypredict
    return np.dot(ydiff.T, ydiff)/X.shape[0]

# returns the Root Mean Squared Error of the model
def getRMSE(X, y, w):
    return np.sqrt(getMSE(X,y,w))[0,0]

def weightCalc(X, y, lamb):
    return np.dot(np.linalg.inv(L2Norm(X)+lamb*np.identity(X.shape[1])), np.dot(X.T, y))

def gradient(X, y, w, lamb):
    return np.dot(X.T, np.dot(X, w)-y)+lamb*w

def doGradientDescent(X, y, lamb): 
    w = np.random.randn(len(X[0]), 1)
    converged = False
    while not converged:
        wnext = w-learningRate*gradient(X, y, w, lamb)
        wdiff = np.absolute(wnext-w)
        try:
            np.where(wdiff > 1e-5)[0][0]
        except IndexError:
            converged = True
        w = wnext
    return w

# returns w = ((X.T*X+lamb*I)^-1)*X.T*y
def getWeights(X, y, gradientDescent=False, lamb=0):
    if gradientDescent:
        return doGradientDescent(X, y, lamb) 
    else:
        return weightCalc(X, y, lamb)

def getKFoldSplit(X, y, k):
    Xs = []
    ys = []
    numDataInSplit = int(len(y)/k)
    for i in range(k):
        startIndex = i*numDataInSplit
        endIndex = startIndex+numDataInSplit
        if endIndex > len(y):
            Xs.append(X[startIndex:])
            ys.append(y[startIndex:])
        else:
            Xs.append(X[startIndex:endIndex])
            ys.append(y[startIndex:endIndex])
    return np.array(Xs), np.array(ys)

def getTrainValidation(Xs, ys, i):
    trainX = []
    trainY = []
    validateX = []
    validateY = []
    for j in range(len(Xs)):
        if j != i:
            trainX.extend(Xs[j])
            trainY.extend(ys[j])
        else:
            validateX.extend(Xs[j])
            validateY.extend(ys[j])
    return np.array(trainX), np.array(trainY), np.array(validateX), np.array(validateY)

def ridgeRegression(X, y, gradientDescent=False):
    Xs, ys = getKFoldSplit(X, y, kFoldSplit)
    bestLambLossW = None
    for i in range(len(Xs)):
        trainX, trainY, validateX, validateY = getTrainValidation(Xs, ys, i)
        lamb = startingLamb
        for j in range(10):
            w = getWeights(trainX, trainY, gradientDescent, 0)
            loss = L2Norm(validateY-np.dot(validateX, w))
            if bestLambLossW is None or loss < bestLambLossW[1]:
                bestLambLossW = [lamb, loss, w]
            lamb = int(lamb/2)
    return bestLambLossW[2], bestLambLossW[0]

def reportErrors(w, X, y):
    print("RMSE = " + str(getRMSE(X, y, w)))

def report(w, trainX, trainY, testX, testY):
    sys.stdout.write("\tTraining ")
    reportErrors(w, trainX, trainY)
    sys.stdout.write("\tTesting ")
    reportErrors(w, testX, testY)

if __name__ == '__main__':
    columnNames, trainX, trainY = getData("http://www.cse.scu.edu/~yfang/coen129/crime-train.txt")
    _, testX, testY = getData("http://www.cse.scu.edu/~yfang/coen129/crime-test.txt")
    trainX = addBiasDummyFeature(trainX)
    testX = addBiasDummyFeature(testX)

    print("Linear Regression:")
    w = getWeights(trainX, trainY)
    report(w, trainX, trainY, testX, testY)

    print("\nRidge Regression:")
    w, lamb = ridgeRegression(trainX, trainY)
    report(w, trainX, trainY, testX, testY)

    print("\nLinear Regression with Gradient Descent:")
    w = getWeights(trainX, trainY, True)
    report(w, trainX, trainY, testX, testY)

    print("\nRidge Regression with Gradient Descent:")
    w, lamb = ridgeRegression(trainX, trainY, True)
    report(w, trainX, trainY, testX, testY)
