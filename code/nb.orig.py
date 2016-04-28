#! /usr/bin/env python3
import pickle
import sys
import csv
import pprint as pp
import math
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

##Global variables
datPath = '../data'
paths = {'stop':datPath+'/stopwords.txt','data':datPath+'/data.txt','cleanData':datPath+'/cleanData.txt'}
#set containing all stop words (words not informative about query)
stopSet = set()
#map emotions we are not modelling to ones we are
emotionMap = {'joy':'joy', 'fear':'anger', 'anger':'anger', 'disgust':'anger', 'sadness':'sadness', 'shame':'sadness', 'guilt':'sadness'}
#emotions being modelled
primaryEmotions = ['joy','anger','sadness']
#set of all important words
lexicon = set()
#number of docs for each term
nks = {}
#number of documents
docCnt = 0
#output
errout = open(os.devnull,'w')
#Gaussian Naive Bayes Classifier
clf = GaussianNB()
#not words
notSwitch = {'arent':'are not','couldnt':'could not','didnt':'did not','doesnt':'does not','dont':'do not','hadnt':'had not','hasnt':'has not','havent':'have not','isnt':'is not','mustnt':'must not','shouldnt':'should not','wasnt':'was not','werent':'were not','wouldnt':'would not'}
################################################################################
############################### DATA SPECIFIC FUNCTIONS #######################
################################################################################

def cleanData(ls):
    return ls.strip().lower().replace('?',' XXQMARKXX').replace('!',' XXEXMARK')

def doNotConsider(emo):
    notConsiderd = set(['guilty','fear'])
    return True if emo in notConsiderd else False

def init(toLower=True):
    """
    Function to fill up some important global variables.
    Should be the first to be called.
    Keyword arguments:
    toLower -- default argument, specifying whether to convert string to lowercase before processing.
    Return:
    testX -- list of all documents 
    testY -- corresponding emotions
    """ 

    global stopSet,paths
    print("init",file=errout)
    (testX,testY) = ([],[])
    with open(paths["stop"]) as sFile:
        for line in sFile:
            stopSet.add(line.strip().lower())

    with open(paths["data"]) as dFile:
        reader = csv.reader(dFile,delimiter='#')
        for row in reader:
            if doNotConsider(row[0]):
                continue
            testY.append(row[0])
            testX.append(cleanData(row[1]))
            #testX.append(row[1].strip().lower().replace('?',' XXQMARKXX').replace('!',' XXEXMARK'))

    return (testX,testY)

def reduceEmotions(ls):
    global emotionMap
    print("reduceEmotions",file=errout)
    ls2 = [emotion for emotion in map(lambda x:emotionMap[x],ls)]
    return ls2

def removeStopWords(ls,addToLexicon=True):
    """
    Compares each word in data to standard stop word list.
    Also builds the lexicon(set of all words) required for model.
    """
    global stopSet,lexicon
    print("removeStopWords",file=errout)
    testX = ['']*len(ls)

    for i in range(len(ls)):
        validLine = []
        for k in notSwitch:
            ls[i] = ls[i].replace(k,notSwitch[k]+" ")

        ls[i] = ls[i].replace(' not ',' not')
        for word in ls[i].split():
            if word not in stopSet:
                if addToLexicon:
                    lexicon.add(word.strip("."))
                validLine.append(word.strip("."))
        testX[i] = " ".join(validLine)
    return testX

def _debug(ls,onlyLen=True):
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

def getDocCountofTerms(testX):
    """
    Function to return list of counts of documents containing a term for each term in the lexicon.
    """
    nks = {}
    for doc in testX:
        for word in set(doc.split()):
            nks[word] = nks[word] + 1 if word in nks else 1
    return nks

def getTermsFrequencyListInDoc(doc):
    fMap = {}
    doc = doc.split()
    for term in doc:
        fMap[term] = fMap[term] + 1 if term in fMap else 1
    return fMap

def getDocumentWeightVector(x):
    global nks,docCnt
    weightVec = []
    normFactor = 10.0
    tf = getTermsFrequencyListInDoc(x)
    for word in lexicon:
        if word not in tf:
            weightVec.append(0.0)
        elif word in nks:
            lg = math.log(docCnt*1.0/nks[word])
            weightVec.append(tf[word]*lg)
            normFactor += (tf[word]*lg)**2
    
    return [x/math.sqrt(normFactor) for x in weightVec]

def getDocumentWeightVectors(ctestX):
    global nks,docCnt
    print("getDocumentWeightVectors",file=errout)
    nks = getDocCountofTerms(ctestX)
    docCnt = len(ctestX)
    return [getDocumentWeightVector(x) for x in ctestX]

def fit(ctestX,ctestY):
    print("getEmotionClassVectors",file=errout)
    emotionWeightMap = dict(zip(primaryEmotions,[[0.0]*len(lexicon) for _ in range(len(primaryEmotions))]))
    docEmotionCnt = {}
    docVectors = getDocumentWeightVectors(ctestX)
    for (i,emotion) in enumerate(ctestY):
        emotionWeightMap[emotion] = [x+y for x,y in zip(emotionWeightMap[emotion],docVectors[i])]
        docEmotionCnt[emotion] = docEmotionCnt[emotion]+1 if emotion in docEmotionCnt else 1

    for e in emotionWeightMap.keys():
        emotionWeightMap[e] = [x/docEmotionCnt[e] for x in emotionWeightMap[e]]
    docVectorsArray=np.asarray(docVectors)
    ctestYArray=np.asarray(ctestY)
    #clf = GaussianNB()
    clf.fit(docVectorsArray,ctestYArray)
    return docVectors

def predict(query):
    #todo:cleanup test data
    query = cleanData(query)
    query = removeStopWords([query],addToLexicon=False)
    query = getDocumentWeightVector(query[0])
    #print(query)

    #args = [(e,sum([x*y for x,y in zip(query,emotions[e])])) for e in emotions]
    return clf.predict((np.asarray(query)).reshape(1,-1))
    #return  max(args, key=lambda x:x[1])[0]


def predictTraining(docVectors,ctestY):
    gauPred=[]
    docVectorsArray=np.asarray(docVectors)
    ctestYArray=np.asarray(ctestY)
    #clf = GaussianNB()
    clf.fit(docVectorsArray,ctestYArray)
    for i in range(len(docVectorsArray)):
        gauPred.append(clf.predict(docVectorsArray[i].reshape(1,-1)))
    gPred=[item for sublist in gauPred for item in sublist] 
    accuracy = dict(zip(primaryEmotions,[dict(zip(primaryEmotions,[0]*len(primaryEmotions))) for _ in range(len(primaryEmotions))]))
    for e in zip(gPred,ctestY):
        accuracy[e[1]][e[0]] += 1

    print(accuracy)
    for k in accuracy:
        print(k,accuracy[k][k]/sum(accuracy[k].values()))
    print("F1 score of Gaussian Naive Bayes")
    print(f1_score(ctestY,gPred,average='macro'))

if __name__ == "__main__":
    (testX,testY) = init()
    ctestX = removeStopWords(testX)
    ctestY = reduceEmotions(testY)
    with open(paths['cleanData'],"w") as cFile:
        for line in ctestX:
            print(line, file=cFile)
    docVectors = fit(ctestX,ctestY)
    #predictTraining(docVectors,ctestY)
    
    #fo = open("../data/test.txt","r")
    #fout = open("../data/test_nb.txt","a")
    #for line in fo:
        #print(predict(line))
    #    fout.write(str(predict(line))+" "+line)

    #query
    query=input("enter query(q to quit)? ")
    while query not in ['Q','q']:
        print(predict(query))
        query=input("enter query(q to quit)? ")
    #print(m['anger'])
    #docVectors = getDocumentWeightVectors(ctestX)
    #print(len(ctestX))
    #print(nks)
    #_debug(zip(testX,ctestX),onlyLen = False)



