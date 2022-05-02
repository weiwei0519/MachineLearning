# coding=UTF-8
# ID3决策树数据集预处理

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

# labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_years', 'marital', 'occupation',
#           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_perweek', 'native_country',
#           'income']

income = {"<=50K": 1, ">50K": 0}

import csv
import pickle
import numpy as np


def data_preprocess(file_name):
    D = []
    labels = []
    with open(file_name, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        line = 0
        # first line is labels
        for row in reader:
            for i, field in enumerate(row):
                row[i] = field.lstrip()
            if line == 0:
                labels = row
            else:
                D.append(row)
            line += 1

    return D, labels  # 返回数据集


if __name__ == '__main__':
    # 训练数据预处理
    file_name = './data/income_trainsets.csv'
    datasets, labels = data_preprocess(file_name)
    print("datasets is maxtrix {0} x {1}".format(len(datasets), len(datasets[0])))
    print(datasets)
    pickle_file = open('./data/trainsets.pkl', 'wb')
    pickle.dump(datasets, pickle_file)
    pickle_file.close()
    pickle_file = open('./data/trainsets_label.pkl', 'wb')
    pickle.dump(labels, pickle_file)
    pickle_file.close()

    file_datasets = open('./data/trainsets_all.txt', 'w', encoding='utf-8')
    for i in range(0, len(datasets)):
        row = datasets[i]
        file_datasets.writelines(str(row) + "\n")
    file_datasets.close()

    # 测试数据预处理
    file_name = './data/income_testsets.csv'
    testsets, labels = data_preprocess(file_name)
    print("testsets is maxtrix {0} x {1}".format(len(testsets), len(testsets[0])))
    print(testsets)
    pickle_file = open('./data/testsets.pkl', 'wb')
    pickle.dump(testsets, pickle_file)
    pickle_file.close()
    pickle_file = open('./data/testsets_label.pkl', 'wb')
    pickle.dump(labels, pickle_file)
    pickle_file.close()

    file_datasets = open('./data/testsets_all.txt', 'w', encoding='utf-8')
    for i in range(0, len(testsets)):
        row = testsets[i]
        file_datasets.writelines(str(row) + "\n")
    file_datasets.close()
