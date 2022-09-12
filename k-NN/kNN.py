# coding=UTF-8
# k近邻法的实现

from datasets.dataset import DataSet
from datasets import DistCalculation as dc
import numpy as np
import pandas as pd
from collections import Counter
from utils.pathutil import PathUtil
from evaluate import evaluate

class KNN:
    X = []  # X数据集
    Y = []  # Y数据集
    d = 0  # 数据集维度
    k = 0  # k个近邻

    def __init__(self, file_name, data_transmit, k=3):
        print("KNN datasets initiation")
        self.datasets = DataSet(file_name, data_transmit)
        self.X = self.datasets.getX()
        self.Y = self.datasets.getY()
        self.d = self.X.shape[1]
        self.k = k

    # 找到Xi_testsets的k个近邻，返回矩阵
    # 距离计算采用欧氏距离
    def getKNN(self, xi_testsets, k):
        X = self.X
        Y = self.Y
        if pd.isnull(k):
            k = self.k
        dist_mat = dc.calc_ed_dist_mat(xi_testsets, X)
        idx = dist_mat.argsort()
        dist_mat = dist_mat[idx]
        Y = Y[idx]
        return dist_mat[0:k], Y[0:k]

    # 用投票法进行分类预测，单个测试样本
    def predict(self, xi_testsets, k):
        X = self.X
        Y = self.Y
        if pd.isnull(k):
            k = self.k
        dist, Y = self.getKNN(xi_testsets, k)
        labels_counts = dict(Counter(Y))
        return max(labels_counts, key=labels_counts.get)

    # 用投票法进行分类预测，多个测试样本
    def predictY(self, X_testsets, k):
        X = self.X
        Y = self.Y
        if pd.isnull(k):
            k = self.k
        Y_predict = []
        for i in range(X_testsets.shape[0]):
            Y_predict.append(self.predict(X_testsets[i, :], k))
        return Y_predict

if __name__ == '__main__':
    pathutil = PathUtil()
    project_folder = pathutil.rootPath
    data_file = project_folder + '/datasets/data/income_trainsets.csv'
    dim = 10
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
        "relationship": {"Wife": 1, "Own-child": 2, "Husband": 3, "Not-in-family": 4, "Other-relative": 5,
                         "Unmarried": 6},
        "race": {"White": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4, "Black": 5},
        "sex": {"Female": 1, "Male": 2},
        "native-country": {"United-States": 1, "Cambodia": 2, "England": 3, "Puerto-Rico": 4, "Canada": 5, "Germany": 6,
                           "Outlying-US(Guam-USVI-etc)": 7, "India": 8, "Japan": 9, "Greece": 10, "South": 11,
                           "China": 12,
                           "Cuba": 13, "Iran": 14, "Honduras": 15, "Philippines": 16, "Italy": 17, "Poland": 18,
                           "Jamaica": 19,
                           "Vietnam": 20, "Mexico": 21, "Portugal": 22, "Ireland": 23, "France": 24,
                           "Dominican-Republic": 25,
                           "Laos": 26, "Ecuador": 27, "Taiwan": 28, "Haiti": 29, "Columbia": 30, "Hungary": 31,
                           "Guatemala": 32,
                           "Nicaragua": 33, "Scotland": 34, "Thailand": 35, "Yugoslavia": 36, "El-Salvador": 37,
                           "Trinadad&Tobago": 38, "Peru": 39, "Hong": 40, "Holand-Netherlands": 41},
        "income": {"<=50K": 1, ">50K": -1}
    }
    knn = KNN(data_file, data_transmit, 3)

    testsets_file = project_folder + '/datasets/data/income_testsets.csv'
    datasets = DataSet(testsets_file, data_transmit)
    X_testsets = datasets.getX()
    Y_testsets = datasets.getY()

    Y_predict = knn.predictY(X_testsets, 3)

    P, R, F1 = evaluate.mixed_mat(Y_predict, Y_testsets)



