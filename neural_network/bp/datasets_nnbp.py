# coding=UTF-8
# 神经网络模型数据预处理

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

age = 0
workclass = {"Private": 1, "Self-emp-not-inc": 2, "Self-emp-inc": 3, "Federal-gov": 4, "Local-gov": 5, "State-gov": 6,
             "Without-pay": 7, "Never-worked": 8}
fnlwgt = 0
education = {"Bachelors": 1, "Some-college": 2, "11th": 3, "HS-grad": 4, "Prof-school": 5, "Assoc-acdm": 6,
             "Assoc-voc": 7, "9th": 8, "7th-8th": 9, "12th": 10, "Masters": 11, "1st-4th": 12, "10th": 13,
             "Doctorate": 14, "5th-6th": 15, "Preschool": 16}
education_years = 0
marital = {"Married-civ-spouse": 1, "Divorced": 2, "Never-married": 3, "Separated": 4, "Widowed": 5,
           "Married-spouse-absent": 6, "Married-AF-spouse": 7}
occupation = {"Tech-support": 1, "Craft-repair": 2, "Other-service": 3, "Sales": 4, "Exec-managerial": 5,
              "Prof-specialty": 6, "Handlers-cleaners": 7, "Machine-op-inspct": 8, "Adm-clerical": 9,
              "Farming-fishing": 10, "Transport-moving": 11, "Priv-house-serv": 12, "Protective-serv": 13,
              "Armed-Forces": 14}
relationship = {"Wife": 1, "Own-child": 2, "Husband": 3, "Not-in-family": 4, "Other-relative": 5, "Unmarried": 6}
race = {"White": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4, "Black": 5}
sex = {"Female": 1, "Male": 2}
capital_gain = 0
capital_loss = 0
hours_perweek = 0
native_country = {"United-States": 1, "Cambodia": 2, "England": 3, "Puerto-Rico": 4, "Canada": 5, "Germany": 6,
                  "Outlying-US(Guam-USVI-etc)": 7, "India": 8, "Japan": 9, "Greece": 10, "South": 11, "China": 12,
                  "Cuba": 13, "Iran": 14, "Honduras": 15, "Philippines": 16, "Italy": 17, "Poland": 18, "Jamaica": 19,
                  "Vietnam": 20, "Mexico": 21, "Portugal": 22, "Ireland": 23, "France": 24, "Dominican-Republic": 25,
                  "Laos": 26, "Ecuador": 27, "Taiwan": 28, "Haiti": 29, "Columbia": 30, "Hungary": 31, "Guatemala": 32,
                  "Nicaragua": 33, "Scotland": 34, "Thailand": 35, "Yugoslavia": 36, "El-Salvador": 37,
                  "Trinadad&Tobago": 38, "Peru": 39, "Hong": 40, "Holand-Netherlands": 41}
income = {"<=50K": 1, ">50K": 0}

import csv
import pickle
import numpy as np


def normalization(x, X):
    """"
    归一化到区间{0,1]
    返回副本
    """
    _range = np.max(X) - np.min(X)
    return (x - np.min(X)) / _range


def standardization(x):
    """"
     将输入x 正态标准化  (x - mu) / sigma   ~   N(0,1)
     返回副本
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma


def data_preprocess(file_name):
    datasets_x = []
    datasets_y = []
    num_labels = 2
    with open(file_name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip first headline.
        line = 0
        for row in reader:
            # print(row)
            line += 1
            for i, field in enumerate(row):
                field = field.lstrip()
                if (field.isdigit()):
                    row[i] = int(row[i])
                else:
                    # row[i] = 'headers[i]'.get(field)
                    exec('''row[i] = {0}.get('{1}')'''.format(headers[i], field))
            # print("line: " + str(line))
            # print(row[:len(row)-2])
            # print(row[len(row)-1:])
            datasets_x.append(row[:len(row) - 1])
            datasets_y.append(row[len(row) - 1:])

        # X = np.mat(datasets_x, dtype='float32')
        # Y = np.mat(datasets_y)
        X = np.array(datasets_x, dtype='float32')
        Y = np.array(datasets_y, dtype='int')

        # 对X矩阵中，超过1的数据，进行归一化处理
        # 考虑sigmoid函数，如果X太小，则无法输出Y达到1
        for j in range(0, X.shape[1]):
            # print("before normalization: X[:,{0}] = {1}".format(j, X[:, j]))
            Xj = X[:, j]
            if np.max(Xj) > 1:
                for i in range(0, X.shape[0]):
                    X[i, j] = normalization(X[i, j], X[:, j])
                # print("after normalization: X[:,{0}] = {1}".format(j, X[:, j]))

    return X, Y  # 返回训练数据集


if __name__ == '__main__':
    # 训练数据预处理
    file_name = './data/income_trainsets.csv'
    X, Y = data_preprocess(file_name)
    pickle_file_X = open('./data/trainsets_X.pkl', 'wb')
    pickle_file_Y = open('./data/trainsets_Y.pkl', 'wb')
    pickle.dump(X, pickle_file_X)
    pickle_file_X.close()
    pickle.dump(Y, pickle_file_Y)
    pickle_file_Y.close()

    print("train datasets X is maxtrix {0} x {1}".format(X.shape[0], X.shape[1]))
    print(X)
    print("train datasets Y is maxtrix {0} x {1}".format(Y.shape[0], Y.shape[1]))
    print(Y)

    file_datasets = open('./data/trainsets_all.txt', 'w', encoding='utf-8')
    for i in range(X.shape[0]):
        row = np.hstack((X[i, :], Y[i, :]))
        file_datasets.writelines(str(row) + "\n")
    file_datasets.close()

    # 测试数据预处理
    file_name = './data/income_testsets.csv'
    X, Y = data_preprocess(file_name)
    pickle_file_X = open('./data/testsets_X.pkl', 'wb')
    pickle_file_Y = open('./data/testsets_Y.pkl', 'wb')
    pickle.dump(X, pickle_file_X)
    pickle_file_X.close()
    pickle.dump(Y, pickle_file_Y)
    pickle_file_Y.close()

    print("test datasets X is maxtrix {0} x {1}".format(X.shape[0], X.shape[1]))
    print(X)
    print("test datasets Y is maxtrix {0} x {1}".format(Y.shape[0], Y.shape[1]))
    print(Y)

    file_datasets = open('./data/testsets_all.txt', 'w', encoding='utf-8')
    for i in range(0, X.shape[0]):
        row = np.hstack((X[i, :], Y[i, :]))
        file_datasets.writelines(str(row) + "\n")
    file_datasets.close()
