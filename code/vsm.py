#! /usr/bin/env python3
import sys
import csv
import pprint as pp
import math
import os
from tfidfhelper import tfidfhelper

class vsm:
    def __init__(self, verbose=False, dat='data.txt'):
        """Constructor creates the helper object to be used throughout the model
        arguments: verbose-print extra info?, dat-specify training data file
        return: none
        """
        self.tf = tfidfhelper(dat=dat)        

    def getEmotionClassVectors(self, ctestX,ctestY):
        """
        Function creates the respective emotion class vectors, populated with the mean weights
        of all the documents belonging to that class.
        arguments: ctestX - input dataset, ctestY - observed data
        """
        emotionWeightMap = dict(zip(self.tf.primaryEmotions,[[0.0]*len(self.tf.lexicon) for _ in range(len(self.tf.primaryEmotions))]))
        docEmotionCnt = {}
        docVectors = self.tf.getDocumentWeightVectors(ctestX)
        #count number of documents for each emotion
        for (i,emotion) in enumerate(ctestY):
            emotionWeightMap[emotion] = [x+y for x,y in zip(emotionWeightMap[emotion],docVectors[i])]
            docEmotionCnt[emotion] = docEmotionCnt[emotion]+1 if emotion in docEmotionCnt else 1

        #normalize the vectors
        for e in emotionWeightMap.keys():
            emotionWeightMap[e] = [x/docEmotionCnt[e] for x in emotionWeightMap[e]]

        return emotionWeightMap

    def predict(self, query):
        """
        Simple function that cleans data, and queries the model for a prediction.
        The predicted emotion is returned to the caller.
        arguments: query - a sentence (document)
        return: predicted emotion
        """
        query = self.tf.cleanData(query)
        query = self.tf.removeStopWords([query],addToLexicon=False)

        qtag = [(word, 1 if word in self.tf.lexicon else 0) for word in query[0].split()]
        print(qtag)
        # print(query)
        query = self.tf.getDocumentWeightVector(query[0])
        #print(query)

        args = [(e,sum([x*y for x,y in zip(query,self.classWts[e])])) for e in self.classWts]
        # print(args)
        return  max(args, key=lambda x:x[1])[0]

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

    def fit(self, verbose=False):
        """
        Function to fit the model to the training data. Performs 3 steps:
        -clean the data i.e. remove stop words etc,
        -print cleaned data to a file
        -compute the emotion class weights
        arguments: verbose - print extra info
        return: none
        """
        print("training vsm...")
        (testX,testY) = self.tf.init()
        ctestX = self.tf.removeStopWords(testX)
        ctestY = self.tf.reduceEmotions(testY)
        with open(self.tf.paths['cleanData'],"w") as cFile:
            for line in ctestX:
                print(line, file=cFile)

        self.classWts = self.getEmotionClassVectors(ctestX,ctestY)
        if verbose:
            self.printTestConfMatrix(ctestX,ctestY)


if __name__ == "__main__":
    obj = vsm()
    obj.fit(verbose=False)
    query=input("enter query(q to quit)? ")
    while query not in ['Q','q']:
        print(obj.predict(query))
        query=input("enter query(q to quit)? ")
