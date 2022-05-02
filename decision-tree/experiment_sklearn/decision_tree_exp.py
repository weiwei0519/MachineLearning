# coding=UTF-8
# 决策树模型数据集预处理

# 数据结构
# 观测字段：age,workclass,fnlwgt,education,education-years,marital,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,natice-country
# 分类预测字段：income

# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

headers = ['age', 'workclass', 'fnlwgt', 'education', 'education_years', 'marital', 'occupation',
           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_perweek', 'native_country',
           'income']

data_transmit = {
    "workclass": {"Private": 1, "Self-emp-not-inc": 2, "Self-emp-inc": 3, "Federal-gov": 4, "Local-gov": 5,
                  "State-gov": 6,
                  "Without-pay": 7, "Never-worked": 8},
    "education": {"Bachelors": 1, "Some-college": 2, "11th": 3, "HS-grad": 4, "Prof-school": 5, "Assoc-acdm": 6,
                  "Assoc-voc": 7, "9th": 8, "7th-8th": 9, "12th": 10, "Masters": 11, "1st-4th": 12, "10th": 13,
                  "Doctorate": 14, "5th-6th": 15, "Preschool": 16},
    "marital": {"Married-civ-spouse": 1, "Divorced": 2, "Never-married": 3, "Separated": 4, "Widowed": 5,
                "Married-spouse-absent": 6, "Married-AF-spouse": 7},
    "occupation": {"Tech-support": 1, "Craft-repair": 2, "Other-service": 3, "Sales": 4, "Exec-managerial": 5,
                   "Prof-specialty": 6, "Handlers-cleaners": 7, "Machine-op-inspct": 8, "Adm-clerical": 9,
                   "Farming-fishing": 10, "Transport-moving": 11, "Priv-house-serv": 12, "Protective-serv": 13,
                   "Armed-Forces": 14},
    "relationship": {"Wife": 1, "Own-child": 2, "Husband": 3, "Not-in-family": 4, "Other-relative": 5, "Unmarried": 6},
    "race": {"White": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4, "Black": 5},
    "sex": {"Female": 1, "Male": 2},
    "native_country": {"United-States": 1, "Cambodia": 2, "England": 3, "Puerto-Rico": 4, "Canada": 5, "Germany": 6,
                       "Outlying-US(Guam-USVI-etc)": 7, "India": 8, "Japan": 9, "Greece": 10, "South": 11, "China": 12,
                       "Cuba": 13, "Iran": 14, "Honduras": 15, "Philippines": 16, "Italy": 17, "Poland": 18,
                       "Jamaica": 19,
                       "Vietnam": 20, "Mexico": 21, "Portugal": 22, "Ireland": 23, "France": 24,
                       "Dominican-Republic": 25,
                       "Laos": 26, "Ecuador": 27, "Taiwan": 28, "Haiti": 29, "Columbia": 30, "Hungary": 31,
                       "Guatemala": 32,
                       "Nicaragua": 33, "Scotland": 34, "Thailand": 35, "Yugoslavia": 36, "El-Salvador": 37,
                       "Trinadad&Tobago": 38, "Peru": 39, "Hong": 40, "Holand-Netherlands": 41},
    "income": {"<=50K": 1, ">50K": 0}
}

import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
import os


def data_preprocess(file_name):
    datasets = pd.read_csv(file_name, sep="\s*,\s*")  # sep="\s*,\s*"去掉空格
    datasets = datasets.replace(data_transmit)
    # datasets = datasets.replace(workclass)
    # datasets = datasets.replace(education)
    # datasets = datasets.replace(marital)
    # datasets = datasets.replace(occupation)
    # datasets = datasets.replace(relationship)
    # datasets = datasets.replace(race)
    # datasets = datasets.replace(sex)
    # datasets = datasets.replace(native_country)
    # datasets = datasets.replace(income)
    print(datasets.head(10))  # 输出前10行看看对不对
    X = datasets.drop(columns='income')
    Y = datasets['income']
    return X, Y


def data_split(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    print("X_train data sample: ")
    print(X_train.head(10))
    print("Y_train data sample: ")
    print(Y_train.head(10))
    print("X_test data sample: ")
    print(X_test.head(10))
    print("Y_test data sample: ")
    print(Y_test.head(10))
    return X_train, X_test, Y_train, Y_test


def model_evaluation(Y_predict, Y_test):
    fpr, tpr, threshold = roc_curve(Y_test, Y_predict)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(fpr, tpr)
    plt.title('ROC曲线')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    auc_score = roc_auc_score(Y_test, Y_predict)
    return auc_score


def decision_tree_display(model, X_train):
    os.environ['PATH'] = os.pathsep + r'D:\Program Files\Graphviz\bin'
    data = export_graphviz(model, out_file='tree.dot', feature_names=X_train.columns, class_names=['<=50K', '>50K'],
                           rounded=True, filled=True)
    with open("tree.dot", encoding='utf-8') as f:
        dot_graph = f.read()
    graph = graphviz.Source(dot_graph)
    graph.render('decision_tree.pdf')


if __name__ == '__main__':
    # 训练数据预处理
    file_name = './data/income_datasets.csv'
    ## 数据预处理，对于字符串进行数据化替换
    print("数据预处理，对于字符串进行数据化替换")
    X, Y = data_preprocess(file_name)

    ## 对数据集进行拆分，20%的数据用作测试集，80%用作训练集
    print("对数据集进行拆分，20%的数据用作测试集，80%用作训练集")
    X_train, X_test, Y_train, Y_test = data_split(X, Y)

    ## 对划分好的数据集进行训练
    print("对划分好的数据集进行训练")
    # radom_state 是一个种子参数，它用来保证每次划分的数据集是一致的。
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X_train, Y_train)

    ## 用测试集来进行测试
    print("用测试集来进行测试")
    Y_predict = model.predict(X_test)
    print("Y_predict sample is")
    print(Y_predict[0:10])

    ## 计算预测准确率
    print("计算预测准确率")
    score = accuracy_score(Y_predict, Y_test)
    print("score = {0}".format(score))

    ## 模型评估，画ROC曲线
    print("模型评估，画ROC曲线")
    auc_score = model_evaluation(Y_predict, Y_test)
    print("AUC score is {0}".format(auc_score))

    ## 计算各个特征的影响程度，给模型的特征排名
    features = X.columns
    importances = model.feature_importances_
    a = pd.DataFrame()
    a['名称'] = features
    a['重要性'] = importances
    a.sort_values('重要性', ascending=False)
    print("数据集中，各个特征的重要程度排名如下：")
    print(a)

    ## 决策树可视化
    decision_tree_display(model, X_train)
