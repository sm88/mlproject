#! /usr/bin/env python3

import sys
import csv
import pprint as pp
import math
import os

class tfidfhelper:
    def __init__(self, verbose=False, dat='data.txt'):
        """
        Constructor initializing some important variables, used throughout.
        arguments: verbose-should extra info be printed, dat=specify alternate training file
        return: none
        """
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

        self.doNotConsider = set(['guilt', 'fear'])

    def cleanData(self, ls):
        """
        Function does basic housekeeping and converts important punctuations to words, so
        that they don't end up being removed due to the stop word list removal.
        arguments: ls-data to clean
        return: cleaned data
        """
        return ls.strip().lower().replace('?',' XXQMARKXX').replace('!',' XXEXMARK')

    def reduceEmotions(self, ls):
        """Function to convert the index based outputs to meaningfull strings. Eg 1->"anger"
        arguments: ls-list of index based emotions
        return: list of emotion strings
        """
        print("reduceEmotions",file=self.errout)
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

        return (testX,testY)

    def removeStopWords(self, ls,addToLexicon=True):
        """
        Compares each word in data to standard stop word list.
        Also builds the lexicon(set of all words) required for model.
        arguments: ls-data to be cleansed
        return: cleansed data
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
        """
        Testing function, used to print vectors of data
        arguments: ls-data
        return: none
        """
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
        arguments: testX - test data
        return: nks - word frequency in entire dataset
        """
        nks = {}
        for doc in testX:
            for word in set(doc.split()):
                nks[word] = nks[word] + 1 if word in nks else 1
        return nks

    def getTermsFrequencyListInDoc(self, doc):
        """Function returning frequency of all words in a single sentence (document)
        arguments: doc - the sentence
        return: fMap - dict of term freq
        """
        fMap = {}
        doc = doc.split()
        for term in doc:
            fMap[term] = fMap[term] + 1 if term in fMap else 1
        return fMap

    def getDocumentWeightVector(self, x):
        """
        Function to create vector of the size of the lexicon populated with weights corresponding
        to the terms in the document.
        arguments: x-a single sentence
        return:weight vector
        """
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
        """
        Function returning all document weight vectors in the entire dataset
        arguments: ctestX - cleaned dataset
        return: none
        """
        print("getDocumentWeightVectors",file=self.errout)
        self.nks = self.getDocCountofTerms(ctestX)
        self.docCnt = len(ctestX)
        return [self.getDocumentWeightVector(x) for x in ctestX]
