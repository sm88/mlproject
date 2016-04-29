#! /usr/bin/env python3
import sys
import csv
import pprint as pp
import math
import os
import numpy as np
from sklearn import svm as sv
from sklearn.metrics import f1_score
from tfidfhelper import tfidfhelper

class svm:
    def __init__(self, verbose=False, dat='data.txt'):
        """Constructor creates the helper object to be used throughout the model
        and initialize the SVM model from sklearn.
        arguments: verbose-print extra info?, dat-specify training data file
        return: none
        """
        self.tf = tfidfhelper(dat=dat)
        self.clf = sv.LinearSVC()

    def printTestConfMatrix(self, ctestX, ctestY):
        """
        Function to print the confusion matrix for the training data.
        arguments: ctestX - training dataset, ctestY - observed data
        return: none
        """
        predY=[]
        for (i,test) in enumerate(ctestX):
            print("test %d\r" % (i+1),end='')
            predY.append(self.predict(test))
        
        accuracy = dict(zip(self.tf.primaryEmotions,[dict(zip(self.tf.primaryEmotions,[0]*len(self.tf.primaryEmotions))) for _ in range(len(self.tf.primaryEmotions))]))
        for e in zip(predY,ctestY):
            accuracy[e[1]][e[0]] += 1

        #print(accuracy)

        #print correctly classified over total values
        for k in accuracy:
            print(k,accuracy[k][k]/sum(accuracy[k].values()))
            print(k,accuracy[k])

    def fit(self,verbose=False):
        """
        Function to fit the model to the training data. Performs 3 steps:
        -clean the data i.e. remove stop words etc,
        -print cleaned data to a file
        -compute the emotion class weights
        arguments: verbose - print extra info
        return: none
        """
        print("training svm...")
        (testX,testY) = self.tf.init()
        ctestX = self.tf.removeStopWords(testX)
        ctestY = self.tf.reduceEmotions(testY)
        with open(self.tf.paths['cleanData'],"w") as cFile:
            for line in ctestX:
                print(line, file=cFile)
        docVectors = self.tf.getDocumentWeightVectors(ctestX)
        
        docVectorsArray=np.asarray(docVectors)
        ctestYArray=np.asarray(ctestY)
        #clf = GaussianNB()
        self.clf.fit(docVectorsArray,ctestYArray)
        if verbose:
            self.printTestConfMatrix(ctestX,ctestY)

    def predict(self, query):
        """
        Simple function that cleans data, and queries the model for a prediction.
        The predicted emotion is returned to the caller.
        arguments: query - a sentence (document)
        return: predicted emotion
        """
        query = self.tf.cleanData(query)
        query = self.tf.removeStopWords([query],addToLexicon=False)
        query = self.tf.getDocumentWeightVector(query[0])
        
        return self.clf.predict((np.asarray(query)).reshape(1,-1))[0]

if __name__ == "__main__":
    obj = svm()
    obj.fit(verbose=True)
    query=input("enter query(q to quit)? ")
    while query not in ['Q','q']:
        print(obj.predict(query))
        query=input("enter query(q to quit)? ")