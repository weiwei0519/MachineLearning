# coding=UTF-8
# 朴素贝叶斯算法的实现

# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import numpy as np


def loadCsv(filename):
    lines = csv.reader(open(filename, "rt"))
    next(lines)     # Skip first headline.
    dataset = list(lines)
    # for i in range(len(dataset)):
    #     dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(float(x) - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def col_attr_summarize(dataset):
    col_attr_count = {}
    col = 0
    dispersed = True
    for attribute in zip(*dataset):
        dispersed = True
        # 判断当前属性列是离散型还是连续型
        for i in range(5):
            dispersed = dispersed and not (str(attribute[random.randint(0, len(attribute))]).isdigit())
        if dispersed:
            # 只有离散属性列才需要计算
            N_i = []
            for xi in attribute:
                if xi not in N_i:
                    N_i.append(xi)
            col_attr_count['col_' + str(col)] = len(N_i)
            col_attr_count['col_' + str(col) + '_options'] = N_i
        col += 1

    return col_attr_count


# {"D":xx
#  "class1": {	"col_i": {"col_value": {"xi":D_cxi}; "col_dispersed": True; "N_i":xx}
# 	            "col_i": {"col_value": {"mean":mean;"stdev":stdev}; "col_dispersed": False}
# 	            ...
#                }
#  "class2": {	"col_i": {"col_value": {"xi":D_cxi}; "col_dispersed": True;  "N_i":xx}
# 	            "col_i": {"col_value": {"mean":mean;"stdev":stdev}; "col_dispersed": False}
# 	            ...
#                }
# }

def nb_model_train(dataset):
    col_nums = len(dataset[0])
    nb_model = {}
    col_attr_count = col_attr_summarize(dataset)
    separated = separateByClass(dataset)
    dispersed = True
    for classValue, classInstances in separated.items():
        col = 0
        class_value = {}
        for attribute in zip(*classInstances):
            column_value = {}
            # 判断当前属性列是离散型还是连续型
            for i in range(5):
                dispersed = dispersed and not (str(attribute[random.randint(0, len(attribute))]).isdigit())
            # 离散型属性列，计算P(xi|c)
            if dispersed:
                # 离散属性计算
                D_cx = {}           # 条件概率
                for xi in attribute:
                    if xi not in D_cx:
                        D_cx[xi] = 1
                    else:
                        D_cx[xi] += 1

                # 计算每个属性的条件概率，并引入拉普拉斯平滑
                P_cx = {}
                for xi, D_cxi in D_cx.items():
                    P_cx[xi] = (D_cxi + 1) / (len(attribute) + col_attr_count['col_' + str(col)])
                # 对当前class中，该属性列不存在的option，D_cxi = 0, 也需要计算P_cx, 用拉普拉斯平滑计算
                for option in col_attr_count['col_' + str(col) + '_options']:
                    if option not in P_cx.keys():
                        P_cx[option] = 1 / (len(attribute) + col_attr_count['col_' + str(col)])
                column_value['col_value'] = P_cx.copy()
                column_value['col_dispersed'] = dispersed

                class_value['col_' + str(col)] = column_value.copy()
            else:
                # mean()函数：计算均值；stdev()函数：计算标准差
                p_cx = {}           # 概率密度
                p_cx['mean'] = mean([float(val) for val in attribute])
                p_cx['stdev'] = stdev([float(val) for val in attribute])
                column_value['col_value'] = p_cx.copy()
                column_value['col_dispersed'] = dispersed

                class_value['col_' + str(col)] = column_value.copy()
            col += 1
            dispersed = True
            if col == col_nums - 1:
                break       # 跳过最后一列class结果列
        nb_model[classValue] = class_value.copy()
    nb_model['D'] = len(dataset)

    return nb_model


def calculateProbability(x, mean, stdev):
    # mean属性均值，stdev属性标准差
    # 机器学习7.3 朴素贝叶斯分类器，式(7.18)，连续属性根据概率密度函数计算。
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# {"D":xx
#  "class1": {	"col_i": {"col_value": {"xi":D_cxi}; "col_dispersed": True; "N_i":xx}
# 	            "col_i": {"col_value": {"mean":mean;"stdev":stdev}; "col_dispersed": False}
# 	            ...
#                }
#  "class2": {	"col_i": {"col_value": {"xi":D_cxi}; "col_dispersed": True;  "N_i":xx}
# 	            "col_i": {"col_value": {"mean":mean;"stdev":stdev}; "col_dispersed": False}
# 	            ...
#                }
# }

def calculateClassProbabilities(nb_model, inputVector):
    probabilities = {}
    for labelName, labelValue in nb_model.items():
        if labelName == 'D':
            D = labelValue
        else:
            probabilities[labelName] = 1
            for i in range(len(labelValue)):
                column = labelValue["col_" + str(i)]
                if column["col_dispersed"]:
                    # 离散列，用概率预测
                    colValue = column["col_value"]
                    if inputVector[i] in colValue:
                        p_xi = colValue[inputVector[i]]
                    else:
                        p_xi = 0        # 如果当前离散值在train数据中没有，说明出现的概率为0
                    probabilities[labelName] *= p_xi
                else:
                    colValue = column["col_value"]
                    mean = colValue['mean']
                    stdev = colValue['stdev']
                    x = float(inputVector[i])  # inputVector是一条测试集，测试集的属性数量与classSummaries是一致的。
                    probabilities[labelName] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(nb_model, inputVector):
    probabilities = calculateClassProbabilities(nb_model, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(nb_model, testSet):
    predictions = []
    for i in range(len(testSet)):
        if i == 2:
            print(i)
        result = predict(nb_model, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    test_labels = []
    for i in range(len(testSet)):
        test_labels.append(testSet[i][-1])
        if testSet[i][-1] == predictions[i]:
            correct += 1
    print("test_class: {0}".format(test_labels[:]))
    print("predictions: {0}".format(predictions[:]))
    return (correct / float(len(testSet))) * 100.0


# pima-indians-diabetes.data.csv的下载地址：
# https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)
def main():
    trainfilename = './data/income_trainsets.csv'
    trainingSet = loadCsv(trainfilename)
    testfilename = './data/income_testsets.csv'
    testSet = loadCsv(testfilename)
    print(('datasets train={0} and test={1} rows').format(len(trainingSet), len(testSet)))
    # prepare model
    nb_model = nb_model_train(trainingSet)
    # test model
    predictions = getPredictions(nb_model, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print(('Accuracy: {0}%').format(accuracy))


main()
