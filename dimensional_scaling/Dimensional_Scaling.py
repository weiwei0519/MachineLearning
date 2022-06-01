# coding=UTF-8
# 降维，西瓜书P225

import numpy as np
import pandas as pd
from datasets.dataset import DataSet
from datasets import DistCalculation as dc
from utils.pathutil import PathUtil


class Dimension:
    X = []  # X数据集
    Y = []  # Y数据集
    d = 0  # 数据集维度
    p = 2  # 闵可夫斯基距离计算的指数
    D = []  # 距离矩阵

    def __init__(self, file_name, data_transmit, p=2):
        print("Dimension initial")
        self.dataset = DataSet(file_name, data_transmit)
        self.X = self.dataset.getX()
        self.Y = self.dataset.getY()
        self.d = self.X.shape[1]
        self.p = p
        self.D = np.zeros((self.X.shape[0], self.X.shape[0]))  # 距离矩阵初始化

    # 获取数据集
    def getDatasets(self):
        return self.X

    # 设置降维后的维度d
    def setDimension(self, d):
        self.d = d

    # MDS 多维缩放算法
    def MDS(self, dim):
        X = self.dataset.datasplit(self.X, 5000)
        m, n = X.shape

        # 计算样本X的距离矩阵
        D = dc.calc_dist_mat(X)

        print("距离矩阵D[{0} x {1}]:".format(D.shape[0], D.shape[1]))
        print(D)

        # 生成降维样本Z的内积矩阵B
        B = np.zeros((D.shape))
        Dist_2 = np.power(D, 2)
        Dist_i_2 = (np.sum(Dist_2, axis=1) / m).reshape(-1, 1)
        Dist_j_2 = np.sum(Dist_2, axis=0) / m
        Sum_Dist_2 = np.sum(np.sum(Dist_2, axis=1), axis=0) / (m * m)
        B = -1 / 2 * (Dist_2 - Dist_i_2 - Dist_j_2 + Sum_Dist_2)
        # for i in range(m):
        #     for j in range(i, m):
        #         # b_ij = -1/2 * (dist_ij*dist_ij - dist_i*dist_i - dist_j*dist_j - dist_*dist_)
        #         dist_i2 = np.sum(np.power(np.ravel(D[i, :]), 2), axis=0) / m
        #         dist_j2 = np.sum(np.power(np.ravel(D[:, j]), 2), axis=0) / m
        #         dist_2 = np.sum(np.sum(np.power(D, 2), axis=1), axis=0) / (m * m)
        #         B[i, j] = -1 / 2 * (D[i, j] * D[i, j] - dist_i2 - dist_j2 + dist_2)
        #         B[j, i] = B[i, j]
        print("生成降维样本Z的内积矩阵B[{0} x {1}]:".format(B.shape[0], B.shape[1]))
        print(B)

        # 计算特征值和特征向量矩阵
        gamma, V = np.linalg.eigh(B.dot(B.T))
        A = np.diag(gamma[-dim:])  # 取后dim列
        V = V[:, -dim:]  # 取后dim列
        Z = V.dot(np.power(A, 1 / 2))
        return Z

    def PCA(self, dim):
        X = self.X
        X_minmax = DataSet.getMinMaxDatasets(self.X)
        Centra_X = DataSet.getCentralizedDatasets(X_minmax)  # 对样本数据集进行中心化
        Covariance = Centra_X.T.dot(Centra_X)  # 计算样本的协方差矩阵
        # 对计算协方差矩阵做特征值分解
        # eigh适用于对称矩阵，eig适用于普通矩阵
        gamma, W = np.linalg.eig(Covariance)
        # 对特征值和特征向量进行排序
        idx = gamma.argsort()[::-1]
        gamma = gamma[idx]
        W = W[:, idx]
        print("特征值：\n{0}".format(gamma))
        print("特征向量矩阵：\n{0}".format(W))
        A = np.diag(gamma[-dim:])
        W = W[:, 0:dim]  # 取后dim列
        print("降维{0}特征值对角矩阵：\n{1}".format(dim, A))
        print("降维{0}特征向量矩阵：\n{1}".format(dim, W))
        return X.dot(W)


if __name__ == '__main__':
    data_file = './data/income_trainsets.csv'
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
    dm = Dimension(data_file, data_transmit)
    Z = dm.MDS(dim)
    print("MDS降维后的样本矩阵Z[{0} X {1}]：".format(Z.shape[0], Z.shape[1]))
    print(Z)

    Z = dm.PCA(dim)
    print("PCA算法降维后的样本矩阵Z[{0} X {1}]：".format(Z.shape[0], Z.shape[1]))
    print(Z)
