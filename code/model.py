#! /usr/bin/env python3
import pickle
import sys
import csv
import pprint as pp
import math

##Global variables
datPath = '../data'
paths = {'stop':datPath+'/stopwords.txt','data':datPath+'/data.txt'}
#set containing all stop words (words not informative about query)
stopSet = set()
#map emotions we are not modelling to ones we are
emotionMap = {'joy':'joy', 'fear':'anger', 'anger':'anger', 'disgust':'anger', 'sadness':'sadness', 'shame':'sadness', 'guilt':'sadness'}
#emotions being modelled
primaryEmotions = ['joy','anger','sadness']
#set of all important words
lexicon = set()

################################################################################
############################### DATA SPECIFIC FUNCTIONS #######################
################################################################################
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
    print("init",file=sys.stderr)
    (testX,testY) = ([],[])
    with open(paths["stop"]) as sFile:
        for line in sFile:
            stopSet.add(line.strip().lower())

    with open(paths["data"]) as dFile:
        reader = csv.reader(dFile,delimiter='#')
        for row in reader:
            testY.append(row[0])
            testX.append(row[1].strip().lower().replace('?',' XXQMARKXX').replace('!',' XXEXMARK'))

    return (testX,testY)

def reduceEmotions(ls):
    global emotionMap
    print("reduceEmotions",file=sys.stderr)
    ls2 = [emotion for emotion in map(lambda x:emotionMap[x],ls)]
    return ls2

def removeStopWords(ls):
    """
    Compares each word in data to standard stop word list.
    Also builds the lexicon(set of all words) required for model.
    """
    global stopSet,lexicon
    print("removeStopWords",file=sys.stderr)
    testX = ['']*len(ls)

    for i in range(len(ls)):
        validLine = []
        for word in ls[i].split():
            if word not in stopSet:
                lexicon.add(word.strip("."))
                validLine.append(word.strip("."))
        testX[i] = " ".join(validLine)
        if testX[i]=="":
            print(i,"\n",ls[i-1],"\n",ls[i+1])
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

def getDocumentWeightVector(x,nks,cnt):
    weightVec = []
    normFactor = 0.0
    tf = getTermsFrequencyListInDoc(x)
    for word in lexicon:
        if word not in tf:
            weightVec.append(0.0)
        else:
            lg = math.log(cnt*1.0/nks[word])
            weightVec.append(tf[word]*lg)
            normFactor += (tf[word]*lg)**2
    
    return [x/math.sqrt(normFactor) for x in weightVec]

def getDocumentWeightVectors(ctestX):
    print("getDocumentWeightVectors",file=sys.stderr)
    nks = getDocCountofTerms(ctestX)
    return [getDocumentWeightVector(x,nks,len(ctestX)) for x in ctestX]

def getEmotionClassVectors(ctestX,ctestY):
    print("getEmotionClassVectors",file=sys.stderr)
    emotionWeightMap = dict(zip(primaryEmotions,[[0.0]*len(lexicon) for _ in range(len(primaryEmotions))]))
    docEmotionCnt = {}
    docVectors = getDocumentWeightVectors(ctestX)
    for (i,emotion) in enumerate(ctestY):
        emotionWeightMap[emotion] = [x+y for x,y in zip(emotionWeightMap[emotion],docVectors[i])]
        docEmotionCnt[emotion] = docEmotionCnt[emotion]+1 if emotion in docEmotionCnt else 1

    for e in emotionWeightMap.keys():
        emotionWeightMap[e] = [x/docEmotionCnt[e] for x in emotionWeightMap[e]]

    return emotionWeightMap

def predict(query, emotions):
    #todo:cleanup test data
    max([sum([x*y for x in zip(query,emotions[e])]) for e in emotions])

if __name__ == "__main__":
    (testX,testY) = init()
    ctestX = removeStopWords(testX)
    ctestY = reduceEmotions(testY)
    #print(ctestX)
    m = getEmotionClassVectors(ctestX,ctestY)
    #print(m['anger'])
    #docVectors = getDocumentWeightVectors(ctestX)
    #print(len(ctestX))
    #print(nks)
    #_debug(zip(testX,ctestX),onlyLen = False)


