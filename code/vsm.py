#! /usr/bin/env python3
import sys
import csv
import pprint as pp
import math
import os
from tfidfhelper import tfidfhelper

class vsm:
    def __init__(self, verbose=False, dat='data.txt'):
        self.tf = tfidfhelper()        

    def getEmotionClassVectors(self, ctestX,ctestY):
        # print("getEmotionClassVectors",file=self.errout)
        emotionWeightMap = dict(zip(self.tf.primaryEmotions,[[0.0]*len(self.tf.lexicon) for _ in range(len(self.tf.primaryEmotions))]))
        docEmotionCnt = {}
        docVectors = self.tf.getDocumentWeightVectors(ctestX)
        for (i,emotion) in enumerate(ctestY):
            emotionWeightMap[emotion] = [x+y for x,y in zip(emotionWeightMap[emotion],docVectors[i])]
            docEmotionCnt[emotion] = docEmotionCnt[emotion]+1 if emotion in docEmotionCnt else 1

        for e in emotionWeightMap.keys():
            emotionWeightMap[e] = [x/docEmotionCnt[e] for x in emotionWeightMap[e]]

        return emotionWeightMap

    def predict(self, query):
    #todo:cleanup test data
        query = self.tf.cleanData(query)
        query = self.tf.removeStopWords([query],addToLexicon=False)
        query = self.tf.getDocumentWeightVector(query[0])
        #print(query)

        args = [(e,sum([x*y for x,y in zip(query,self.classWts[e])])) for e in self.classWts]
        return  max(args, key=lambda x:x[1])[0]

    def printTestConfMatrix(self, ctestX, ctestY):
        #training confusion matrix
        predY=[]
        for (i,test) in enumerate(ctestX):
            print("test %d\r" % (i+1),end='')
            predY.append(self.predict(test))
        
        accuracy = dict(zip(self.primaryEmotions,[dict(zip(self.primaryEmotions,[0]*len(self.primaryEmotions))) for _ in range(len(self.primaryEmotions))]))
        for e in zip(predY,ctestY):
            accuracy[e[1]][e[0]] += 1

        print(accuracy)

        for k in accuracy:
            print(k,accuracy[k][k]/sum(accuracy[k].values()))

    def fit(self, verbose=False):
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
    obj = vsm(verbose=True)
    obj.fit(verbose=False)
    query=input("enter query(q to quit)? ")
    while query not in ['Q','q']:
        print(obj.predict(query))
        query=input("enter query(q to quit)? ")
