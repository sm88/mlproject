#! /usr/bin/env python3
import sys
import csv
import pprint as pp
import math
import os

class vsm:
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

    def cleanData(self, ls):
        return ls.strip().lower().replace('?',' XXQMARKXX').replace('!',' XXEXMARK')

    def reduceEmotions(self, ls):
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

    def getEmotionClassVectors(self, ctestX,ctestY):
        print("getEmotionClassVectors",file=self.errout)
        emotionWeightMap = dict(zip(self.primaryEmotions,[[0.0]*len(self.lexicon) for _ in range(len(self.primaryEmotions))]))
        docEmotionCnt = {}
        docVectors = self.getDocumentWeightVectors(ctestX)
        for (i,emotion) in enumerate(ctestY):
            emotionWeightMap[emotion] = [x+y for x,y in zip(emotionWeightMap[emotion],docVectors[i])]
            docEmotionCnt[emotion] = docEmotionCnt[emotion]+1 if emotion in docEmotionCnt else 1

        for e in emotionWeightMap.keys():
            emotionWeightMap[e] = [x/docEmotionCnt[e] for x in emotionWeightMap[e]]

        return emotionWeightMap

    def predict(self, query):
    #todo:cleanup test data
        query = self.cleanData(query)
        query = self.removeStopWords([query],addToLexicon=False)
        query = self.getDocumentWeightVector(query[0])
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
        (testX,testY) = self.init()
        ctestX = self.removeStopWords(testX)
        ctestY = self.reduceEmotions(testY)
        with open(self.paths['cleanData'],"w") as cFile:
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
