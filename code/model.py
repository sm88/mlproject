#! /usr/bin/env python3
import pickle
import sys
import csv
import pprint as pp

##Global variables
datPath = '../data'
paths = {'stop':datPath+'/stopwords.txt','data':datPath+'/data.txt'}
#set containing all stop words (words not informative about query)
stopSet = set()
#map emotions we are not modelling to ones we are
emotionMap = {'joy':'joy', 'fear':'anger', 'anger':'anger', 'disgust':'anger', 'sadness':'sadness', 'shame':'sadness', 'guilt':'sadness'}
#emotions being modelled
primaryEmotion = ['joy','anger','sadness']

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
    (testX,testY) = ([],[])
    with open(paths["stop"]) as sFile:
        for line in sFile:
            stopSet.add(line.strip().lower())

    with open(paths["data"]) as dFile:
        reader = csv.reader(dFile,delimiter='#',quotechar='"')
        for row in reader:
            testY.append(row[0])
            testX.append(row[1].strip().lower())

    return (testX,testY)

def reduceEmotions(ls):
    global emotionMap
    ls2 = [emotion for emotion in map(lambda x:emotionMap[x],ls)]
    return ls2

def removeStopWords(ls):
    global stopSet
    testX = ['']*len(ls)

    for i in range(len(ls)):
        validLine = []
        for word in ls[i].split():
            if word not in stopSet:
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



if __name__ == "__main__":
    (testX,testY) = init()
    ctestX = removeStopWords(testX)
    ctestY = reduceEmotions(testY)
    #_debug(zip(testX,ctestX),onlyLen = False)


