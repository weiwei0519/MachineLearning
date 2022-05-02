# coding=UTF-8
# k近邻法的实现：kd tree

from math import sqrt
from random import randint


##Generate KD tree
def createTree(dataSet, layer=0, feature=2):
    length = len(dataSet)
    dataSetCopy = dataSet[:]
    featureNum = layer % feature
    dataSetCopy.sort(key=lambda x: x[featureNum])
    layer += 1
    if length == 0:
        return None
    elif length == 1:
        return {'Value': dataSet[0], 'Layer': layer, 'feature': featureNum, 'Left': None, 'Right': None}
    elif length != 1:
        midNum = length // 2
        dataSetLeft = dataSetCopy[:midNum]
        dataSetRight = dataSetCopy[midNum + 1:]
        return {'Value': dataSetCopy[midNum], 'Layer': layer, 'feature': featureNum,
                'Left': createTree(dataSetLeft, layer)
            , 'Right': createTree(dataSetRight, layer)}


# calculate distance，计算欧式距离
def calDistance(sourcePoint, targetPoint):
    length = len(targetPoint)
    sum = 0.0
    for i in range(length):
        sum += (sourcePoint[i] - targetPoint[i]) ** 2
    sum = sqrt(sum)
    return sum


# DFS algorithm，deep first search 算法
def dfs(kdTree, target, tracklist=[]):
    tracklistCopy = tracklist[:]
    if not kdTree:
        return None, tracklistCopy
    elif not kdTree['Left']:
        tracklistCopy.append(kdTree['Value'])
        return kdTree['Value'], tracklistCopy
    elif kdTree['Left']:
        pointValue = kdTree['Value']
        feature = kdTree['feature']
        tracklistCopy.append(pointValue)
        # return kdTree['Value'], tracklistCopy
        if target[feature] <= pointValue[feature]:
            return dfs(kdTree['Left'], target, tracklistCopy)
        elif target[feature] > pointValue[feature]:
            return dfs(kdTree['Right'], target, tracklistCopy)


# A function use to find a point in KDtree
def findPoint(Tree, value):
    if Tree != None and Tree['Value'] == value:
        return Tree
    else:
        if Tree['Left'] != None:
            return findPoint(Tree['Left'], value) or findPoint(Tree['Right'], value)


# KDtree search algorithm
def kdTreeSearch(tracklist, target, usedPoint=[], minDistance=float('inf'), minDistancePoint=None):
    tracklistCopy = tracklist[:]
    usedPointCopy = usedPoint[:]

    if not minDistancePoint:
        minDistancePoint = tracklistCopy[-1]

    if len(tracklistCopy) == 1:
        return minDistancePoint
    else:
        point = findPoint(kdTree, tracklist[-1])

        if calDistance(point['Value'], target) < minDistance:
            minDistance = calDistance(point['Value'], target)
            minDistancePoint = point['Value']
        fatherPoint = findPoint(kdTree, tracklistCopy[-2])
        fatherPointval = fatherPoint['Value']
        fatherPointfea = fatherPoint['feature']

        if calDistance(fatherPoint['Value'], target) < minDistance:
            minDistance = calDistance(fatherPoint['Value'], target)
            minDistancePoint = fatherPoint['Value']

        if point == fatherPoint['Left']:
            anotherPoint = fatherPoint['Right']
        elif point == fatherPoint['Right']:
            anotherPoint = fatherPoint['Left']

        if (anotherPoint == None or anotherPoint['Value'] in usedPointCopy or
                abs(fatherPointval[fatherPointfea] - target[fatherPointfea]) > minDistance):
            usedPoint = tracklistCopy.pop()
            usedPointCopy.append(usedPoint)
            return kdTreeSearch(tracklistCopy, target, usedPointCopy, minDistance, minDistancePoint)
        else:
            usedPoint = tracklistCopy.pop()
            usedPointCopy.append(usedPoint)
            subvalue, subtrackList = dfs(anotherPoint, target)
            tracklistCopy.extend(subtrackList)
            return kdTreeSearch(tracklistCopy, target, usedPointCopy, minDistance, minDistancePoint)


trainingSet = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
kdTree = createTree(trainingSet)
target = eval(input('Input target point:'))
value, trackList = dfs(kdTree, target)
nnPoint = kdTreeSearch(trackList, target)
print(nnPoint)


