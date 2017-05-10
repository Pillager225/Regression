#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
import math

learningRate = 1e-6
kFoldSplit = 5
startingLamb = int(400)
numberOfLambs = 100
convergenceDesc = 1e-10

# parses data and splits into table header, data, and labels
def getData(path):
    raw = np.array(pd.read_csv(path, delimiter='\t', header=-1))
    columnNames = raw[0,:]
    X = np.array(raw[1:,1:], dtype=np.float32)
    y = np.array(raw[1:,1], dtype=np.float32)[np.newaxis].T
    return columnNames, X, y

# returs an array that is X with another column vector of 1s
def addBiasDummyFeature(X):
    newX = np.zeros((len(X),len(X[0])+1), dtype=np.float32)
    newX[:,:-1] = X
    newX[0:,len(X[0])] = np.ones(X.shape[0])
    return newX

def L2Norm(V):
    return np.dot(V.T, V)

# ((y-X*w)^2)/n
def getMSE(X, y, w):
    ypredict = np.dot(X, w)
    ydiff = y-ypredict
    return L2Norm(ydiff)[0,0]/X.shape[0]

# returns the Root Mean Squared Error of the model
def getRMSE(X, y, w):
    return np.sqrt(getMSE(X,y,w))

# returns ((X.T*X+lamb*I)^-1)*X.T*y which is  px1
def weightCalc(X, y, lamb):
    return np.dot(np.linalg.inv(L2Norm(X)+lamb*np.identity(X.shape[1], dtype=np.float32)), np.dot(X.T, y))

# returns a px1 vector that represents the gradient of the loss function with respect to w
def gradient(X, y, w, lamb):
    return np.dot(X.T, np.dot(X, w)-y)+lamb*w

def doGradientDescent(X, y, lamb): 
    w = np.random.randn(len(X[0]), 1)
    converged = False
    while not converged:
        wnext = w-learningRate*gradient(X, y, w, lamb)
        wdiff = L2Norm(wnext-w)
        if(wdiff < convergenceDesc):
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
    # best stores the best lamb and error
    # best = (lamb, error)
    best = None
    lamb = startingLamb
    for j in range(numberOfLambs):
        RMSEError = 0.0
        for i in range(kFoldSplit):
            trainX, trainY, validateX, validateY = getTrainValidation(Xs, ys, i)
            w = getWeights(trainX, trainY, gradientDescent, lamb)
            RMSEError += getRMSE(trainX, trainY, w)
        RMSEError /= kFoldSplit
        if best is None or RMSEError < best[1]:
            best = (lamb, RMSEError)
        lamb /= 2
    w = getWeights(X, y, gradientDescent, best[0])
    return w, best[0]

def reportErrors(w, X, y):
    print("RMSE = " + str(getRMSE(X, y, w)))

if __name__ == '__main__':
    columnNames, trainX, trainY = getData("http://www.cse.scu.edu/~yfang/coen129/crime-train.txt")
    _, testX, testY = getData("http://www.cse.scu.edu/~yfang/coen129/crime-test.txt")
    trainX = addBiasDummyFeature(trainX)
    testX = addBiasDummyFeature(testX)

    print("Linear Regression:")
    w = getWeights(trainX, trainY)
    sys.stdout.write("\tTraining ")
    reportErrors(w, trainX, trainY)
    sys.stdout.write("\tTesting ")
    reportErrors(w, testX, testY)

    print("\nRidge Regression:")
    rw, lamb = ridgeRegression(trainX, trainY)
    print(L2Norm(w-rw))
    sys.stdout.write("\tTesting ")
    reportErrors(rw, testX, testY)

    print("\nLinear Regression with Gradient Descent:")
    lgw = getWeights(trainX, trainY, True)
    print(L2Norm(w-lgw))
    sys.stdout.write("\tTraining ")
    reportErrors(lgw, trainX, trainY)
    sys.stdout.write("\tTesting ")
    reportErrors(lgw, testX, testY)

    print("\nRidge Regression with Gradient Descent:")
    grw, lamb = ridgeRegression(trainX, trainY, True)
    print(L2Norm(rw-grw))
    sys.stdout.write("\tTesting ")
    reportErrors(grw, testX, testY)
