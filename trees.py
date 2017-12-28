# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:52:07 2017
decision tree
@author: Administrator
"""
import operator
from math import log

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# Function to calculate the Shannon entropy of a dataset
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
#把数据放到字典labelCount中，然后按key值计算类数
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:#calculate the entropy
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2（累加）
    return shannonEnt
#The higher the entropy,the more mixed up the data is.
dataSet, labels=createDataSet() 
print('calcShannonEnt:',calcShannonEnt(dataSet))
'''
Dataset splitting on a given feature
参数：axis代表按哪个特征分类，value代表分出来显示的是axis中的那种特点
dataSet:the dataset we'll split
axis:the feature we'll split on
value: the value of the feature to return
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
print('splitDataSet:',splitDataSet(dataSet, 0, 1))

#Choosing the best feature to split on
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)      #get a set of unique values
        '''
        #set:Build an unordered collection of unique elements.
        #把featList的元素无序输出，且每个元素只输出一次
        就本程序的而言，就是把每一种特征的具体特点放在uniqueVals这个列表中        
        '''
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)#调用上面自己编写的函数
            #把dataSet中的数据按第i个特征分类，并把第i个特征中具有value特点的数据输出
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature  #返回按哪个特征分类   #returns an integer
print('chooseBestFeatureToSplit:',chooseBestFeatureToSplit(dataSet))

#majority vote
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # classCount.items()被排序的字典类型的view。
    #key:制订排序方式。operator模块提供的itemgetter函数用于获取对象的某维的数据
    # resverse：True 降序（descending），False 升序（ascending）
    return sortedClassCount[0][0]#return the class that occurs with the greatest frequency.

# tree-building code
def createTree(dataSet,labels):
    #labels:The list of labels contains a label for each of the features in the dataset.
    classList = [example[-1] for example in dataSet]
    #ClassList:a list of all the class labels in our dataset.
    if classList.count(classList[0]) == len(classList): 
    #数一下classList里面有几个（classList[0]),如果和classList一样，则此分类全部相同
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree
print('createTree:',createTree(dataSet,labels))
