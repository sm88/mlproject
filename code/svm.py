#! /usr/bin/env python3
import sys
import csv
import pprint as pp
import math
import os
import numpy as np
from sklearn import svm as sv
from sklearn.metrics import f1_score

class svm:
    def __init__(self, verbose=False, dat='data.txt'):
        self.datPath = '../data'
        self.paths = {'stop':self.datPath+'/stopwords.txt','data':self.datPath+'/'+dat,'cleanData':self.datPath+'/cleanData.txt'}
        #set containing all stop words (words not informative about query)
        self.stopSet = set()
        #map emotions we are not modelling to ones we are
        self.emotionMap = {'joy':'joy', 'fear':'anger', 'anger':'anger', 'disgust':'anger', 'sadness':'sadness', 'shame':'sadness', 'guilt':'sadness'}
        #emotions being modelled
        self.primaryEmotions = ['joy','anger','sadness']
        #set of all important words
        self.lexicon = set()
        #number of docs for each term
        self.nks = {}
        #number of documents
        self.docCnt = 0
        #output        
        self.errout = sys.stderr if verbose else open(os.devnull,'w')
        #not words
        self.notSwitch = {'arent':'are not','couldnt':'could not','didnt':'did not','doesnt':'does not','dont':'do not','hadnt':'had not','hasnt':'has not','havent':'have not','isnt':'is not','mustnt':'must not','shouldnt':'should not','wasnt':'was not','werent':'were not','wouldnt':'would not'}

        self.doNotConsider = set(['guilty', 'fear'])

        self.clf = sv.LinearSVC()

    def cleanData(self, ls):
        return ls.strip().lower().replace('?',' XXQMARKXX').replace('!',' XXEXMARK')

    def reduceEmotions(self, ls):
        # print("reduceEmotions",file=self.errout)
        ls2 = [emotion for emotion in map(lambda x:self.emotionMap[x],ls)]
        return ls2

    def init(self, toLower=True):
        """
        Function to fill up some important global variables.
        Should be the first to be called.

        Keyword arguments:
        toLower -- default argument, specifying whether to convert string to lowercase before processing.

        Return:
        testX -- list of all documents 
        testY -- corresponding emotions
        """ 
        print("init",file=self.errout)
        (testX,testY) = ([],[])
        with open(self.paths["stop"]) as sFile:
            for line in sFile:
                self.stopSet.add(line.strip().lower())

        with open(self.paths["data"]) as dFile:
            reader = csv.reader(dFile,delimiter='#')
            for row in reader:
                if row[0] in self.doNotConsider:
                    continue
                testY.append(row[0])
                testX.append(self.cleanData(row[1]))
                #testX.append(row[1].strip().lower().replace('?',' XXQMARKXX').replace('!',' XXEXMARK'))

        return (testX,testY)

    def removeStopWords(self, ls,addToLexicon=True):
        """
        Compares each word in data to standard stop word list.
        Also builds the lexicon(set of all words) required for model.
        """
        #print("removeStopWords",file=self.errout)
        testX = ['']*len(ls)

        for i in range(len(ls)):
            validLine = []
            for k in self.notSwitch:
                ls[i] = ls[i].replace(k,self.notSwitch[k]+" ")

            ls[i] = ls[i].replace(' not ',' not')
            for word in ls[i].split():
                if word not in self.stopSet:
                    if addToLexicon:
                        self.lexicon.add(word.strip("."))
                    validLine.append(word.strip("."))
            testX[i] = " ".join(validLine)
        return testX

    def _debug(self, ls,onlyLen=True):
        for row in ls:
            if onlyLen:
                for item in row:
                    print(len(item),end=' ')
                print()
            else:
                for row in ls:
                    pp.pprint(row)

################################################################################
############################### MODEL SPECIFIC FUNCTIONS #######################
################################################################################

    def getDocCountofTerms(self, testX):
        """
        Function to return list of counts of documents containing a term for each term in the lexicon.
        """
        nks = {}
        for doc in testX:
            for word in set(doc.split()):
                nks[word] = nks[word] + 1 if word in nks else 1
        return nks

    def getTermsFrequencyListInDoc(self, doc):
        fMap = {}
        doc = doc.split()
        for term in doc:
            fMap[term] = fMap[term] + 1 if term in fMap else 1
        return fMap

    def getDocumentWeightVector(self, x):
        weightVec = []
        normFactor = 10.0
        tf = self.getTermsFrequencyListInDoc(x)
        for word in self.lexicon:
            if word not in tf:
                weightVec.append(0.0)
            elif word in self.nks:
                lg = math.log(self.docCnt*1.0/self.nks[word])
                weightVec.append(tf[word]*lg)
                normFactor += (tf[word]*lg)**2
        
        return [x/math.sqrt(normFactor) for x in weightVec]

    def getDocumentWeightVectors(self, ctestX):
        print("getDocumentWeightVectors",file=self.errout)
        self.nks = self.getDocCountofTerms(ctestX)
        self.docCnt = len(ctestX)
        return [self.getDocumentWeightVector(x) for x in ctestX]

    def fit(self):
        print("training svm...")
        (testX,testY) = self.init()
        ctestX = self.removeStopWords(testX)
        ctestY = self.reduceEmotions(testY)
        with open(self.paths['cleanData'],"w") as cFile:
            for line in ctestX:
                print(line, file=cFile)
        docVectors = self.getDocumentWeightVectors(ctestX)
        
        docVectorsArray=np.asarray(docVectors)
        ctestYArray=np.asarray(ctestY)
        #clf = GaussianNB()
        self.clf.fit(docVectorsArray,ctestYArray)

    def predict(self, query):
        #todo:cleanup test data
        query = self.cleanData(query)
        query = self.removeStopWords([query],addToLexicon=False)
        query = self.getDocumentWeightVector(query[0])
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