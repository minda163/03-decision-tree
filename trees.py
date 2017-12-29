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


def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
#作用：来判断一个对象是否是一个已知的类型。 
#其第一个参数（object）为对象，第二个参数（type）为类型名(int...)或类型名的一个列表([int,list,float]是一个列表)。
#其返回值为布尔型（True or flase）。
#若对象的类型与参数二的类型相同则返回True。若参数二为一个元组，则若对象类型与元组中类型名之一相同即返回True。
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel
    
myTree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
dataSet, labels=createDataSet() 
print('classify:',classify(myTree,labels,[1,0]))

#Methods for persisting the decision tree with pickle
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
'''python的pickle模块实现了基本的数据序列和反序列化。
通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储；
通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。　　
基本接口：pickle.dump(obj, file, [,protocol])
注解：将对象obj保存到文件file中去。
　　　protocol为序列化使用的协议版本，
   0：ASCII协议，所序列化的对象使用可打印的ASCII码表示；
   1：老式的二进制协议；
   2：2.3版本引入的新二进制协议，较以前的更高效。
   其中协议0和1兼容老版本的python。protocol默认值为0。
　　　file：对象保存到的类文件对象。
   file必须有write()接口，file可以是一个以'w'方式打开的文件
   或者一个StringIO对象或者其他任何实现write()接口的对象。
   如果protocol>=1，文件对象需要是二进制模式打开的。
　　pickle.load(file)
　　注解：从file中读取一个字符串，并将它重构为原来的python对象。
　　file:类文件对象，有read()和readline()接口。'''
    
storeTree(myTree,'classifierStorage.txt')
print(grabTree('classifierStorage.txt'))

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLabels)
print(lensesTree)
import treePlotter
treePlotter.createPlot(lensesTree)