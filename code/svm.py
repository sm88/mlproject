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
        self.tf = tfidfhelper()
        self.clf = sv.LinearSVC()

    def fit(self):
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

    def predict(self, query):
        #todo:cleanup test data
        query = self.tf.cleanData(query)
        query = self.tf.removeStopWords([query],addToLexicon=False)
        query = self.tf.getDocumentWeightVector(query[0])
        #print(query)

        #args = [(e,sum([x*y for x,y in zip(query,emotions[e])])) for e in emotions]
        return self.clf.predict((np.asarray(query)).reshape(1,-1))[0]

if __name__ == "__main__":
    obj = svm()
    obj.fit()
    query=input("enter query(q to quit)? ")
    while query not in ['Q','q']:
        print(obj.predict(query))
        query=input("enter query(q to quit)? ")