# coding=UTF-8
# 基于ID3算法构造decision tree

from math import log
import operator
import pickle
import sys

sys.path.append("..")
import treePlotter  # 决策树可视化

# 数据集采用adult_income
labels = []
origal_labels = []

##### 计算信息熵 ######
def calcEntropy(D):
    numEntries = len(D)  # 样本数
    labelCounts = {}  # 创建一个数据字典：key是目标分类的类别，value是属于该类别的样本个数
    for featVec in D:  # 遍历整个数据集，每次取一行
        currentLabel = featVec[-1]  # 取该行最后一列的值
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    entropy = 0.0  # 初始化信息熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        entropy -= prob * log(prob, 2)  # log base 2  计算信息熵
    return entropy


##### 按给定的特征划分数据 #########
def splitDataSet(D, axis, value):
    # axis是dataSet数据集下要进行特征划分的列号，value是该列下某个特征值
    retDataSet = []
    for featVec in D:  # 遍历数据集，并抽取按axis的当前value特征进划分的数据集(不包括axis列的值)
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    # print retDataSet
    return retDataSet


##### 计算信息增益 ######
## D 数据集，a 属性取值，label属性名
def calcGain(D, a_values, a_label):
    ent_D = calcEntropy(D)  # 计算入参数据集的信息熵
    ent_Dv = 0.0
    # Gain(D, a) = Ent(D) - sum(|Dv|/|D| * Ent(Dv))   Dv为属性a在v取值下的数据子集，|D|为数据集的数据个数
    for v in a_values:  # 计算每种划分方式的信息熵
        Dv = splitDataSet(D, labels.index(a_label), v)
        prob = len(Dv) / float(len(D))
        ent_Dv += prob * calcEntropy(Dv)
    Gain_D_a = ent_D - ent_Dv
    # print("Gain(D<{0}x{1}>, {2}) = {3}".format(len(D), len(D[0]), a_label, Gain_D_a))

    return Gain_D_a


##### 选取当前数据集下，用于划分数据集的最优特征
def chooseBestFeatureToSplit(D):
    numFeatures = len(D[0]) - 1     # 获取当前数据集的特征个数，最后一列是分类标签
    best_i = -1                     # 最优的特征列号
    best_ai = ""                    # 最优的特征label
    max_Gain_D = -1.0               # 最优信息增益，信息增益最小值为0
    for i in range(numFeatures):
        a_List = [column[i] for column in D]  # 获取数据集中当前特征下的所有值
        a_values = set(a_List)  # 获取当前特征值
        Gain_D_a = calcGain(D, a_values, labels[i])

        if (Gain_D_a > max_Gain_D):  # 比较每个特征的信息增益，只要最好的信息增益
            max_Gain_D = Gain_D_a  # if better than current best, set to best
            best_ai = labels[i]
            best_i = i

    print("best_ai is {0}, best_i is {1}, max_Gain_D = {2}".format(best_ai, best_i, max_Gain_D))
    return best_ai, best_i, max_Gain_D


#####该函数使用分类名称的列表，然后创建键值为classList中唯一值的数据字典。字典
#####对象的存储了classList中每个类标签出现的频率。最后利用operator操作键值排序字典，
#####并返回出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for clazz in classList:
        if clazz not in classCount.keys():
            classCount[clazz] = 0
        classCount[clazz] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


##### 生成决策树主方法  机器学习：第4章决策树 Page74 图4.2
def createTree(D):
    classList = []
    print("current labels is {0}".format(labels))
    if len(labels) == 1:
        return
    for row in D:
        classList.append(row[-1])

    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 当类别完全相同时则停止继续划分，直接返回该类的标签
    if len(D[0]) == 1:  ##遍历完所有的特征时，仍然不能将数据集划分成仅包含唯一类别的分组 dataSet
        return majorityCnt(classList)  # 由于无法简单的返回唯一的类标签，这里就返回出现次数最多的类别作为返回值

    best_ai, best_i, max_Gain_D = chooseBestFeatureToSplit(D)  # 获取最好的分类特征索引

    # 这里直接使用字典变量来存储树信息，这对于绘制树形图很重要。
    myTree = {best_ai: {}}  # 当前数据集选取最好的特征存储在bestFeat中
    del (labels[best_i])    # 删除已经在选取的特征
    best_a_list = [example[best_i] for example in D]
    best_a_values = set(best_a_list)
    for value in best_a_values:
        myTree[best_ai][value] = createTree(splitDataSet(D, best_i, value))
    print("ID3 Tree is: ")
    print(myTree)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


if __name__ == '__main__':
    # model test, load testsets.
    pickle_file_train = open('./data/trainsets.pkl', 'rb+')
    datasets = pickle.load(pickle_file_train)
    print("datasets is maxtrix {0} x {1}".format(len(datasets), len(datasets[0])))
    pickle_file_train.close()

    pickle_file_label = open('./data/trainsets_label.pkl', 'rb+')
    origal_labels = pickle.load(pickle_file_label)
    labels = origal_labels
    pickle_file_label.close()

    id3_tree = createTree(datasets)
    print(id3_tree)
    treePlotter.createPlot(id3_tree)
