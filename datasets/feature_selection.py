# coding=UTF-8
# 特征选择，西瓜书P249 ~ 254

from datasets import DistCalculation as dc
from datasets.dataset import DataSet
import numpy as np
from collections import Counter

np.set_printoptions(suppress=True)  # suppress=True 取消科学记数法


class FeatureSelection:
    X = []  # X数据集
    Y = []  # Y数据集
    d = 0  # 数据集维度
    p = 2  # 闵可夫斯基距离计算的指数

    def __init__(self, file_name, data_transmit, p=2):
        print("Feature Selection initial")
        self.dataset = DataSet(file_name, data_transmit)

    # 样本与猜对近邻和猜错近邻的diff计算
    @staticmethod
    def diff(xa, xb):
        if xa.shape != xb.shape:
            print("Xa and Xb have no same dimension! Please check!")
            return
        return np.abs(xa - xb)

    # 直接用矩阵进行计算，提高计算效率
    @staticmethod
    def Diff(Xa, Xb):
        if Xa.shape != Xb.shape:
            print("Xa and Xb have no same dimension! Please check!")
            return
        return np.abs(Xa - Xb)

    # 计算X与猜中近邻矩阵，猜错近邻矩阵的统计分量贡献
    def calc_delta(self, X, X_nh, X_nm):
        if X.shape != X_nh.shape or X.shape != X_nm.shape:
            print("X, X_nh and X_nm have no same dimension! Please check!")
            return
        diff = -1 * np.power(self.Diff(X, X_nh), 2) + np.power(self.Diff(X, X_nm), 2)
        delta = np.sum(diff, axis=0)
        return delta

    # 计算X与猜中近邻矩阵，猜错近邻矩阵的统计分量贡献，针对多分类情况
    def calc_delta_F(self, X, X_nh, X_nm, Y):
        if X.shape != X_nh.shape or X.shape != X_nm.shape:
            print("X, X_nh and X_nm have no same dimension! Please check!")
            return
        labels = dict(Counter(Y))
        plables = {}
        for label in labels.key():
            plables[label] = labels[label] / Y.shape[0]
        diff_nh = -1 * np.power(self.diff(X, X_nh), 2)
        diff_nm = np.zeros((X.shape))
        for i in range(X.shape[0]):
            for label in labels.key():
                if Y[i] == label:
                    pl = 0
                else:
                    pl = plables[label]
                diff_nm[i, :] += pl * np.power(self.diff(X[i, :], X_nm[label][i, :]))
        diff = diff_nh + diff_nm
        delta = np.sum(diff, axis=0)
        return delta

    # 二分类分类问题
    def Relief(self):
        print("start Relief calculation")
        # self.dataset.datasplit(5000)
        X = self.dataset.getMinMaxDatasets()
        Y = self.dataset.getY()
        X_nh = np.zeros((X.shape))
        X_nm = np.zeros((X.shape))
        print("开始计算样本X的猜中近邻矩阵和猜错近邻矩阵")
        for i in range(X.shape[0]):
            # 计算X的猜中近邻矩阵与猜错近邻矩阵
            print(i)
            xi_nh, xi_nm = dc.getNHM(X[i, :], Y[i], X, Y)
            X_nh[i, :] = xi_nh
            X_nm[i, :] = xi_nm
            print("猜中近邻：{0}".format(xi_nh))
            print("猜错近邻：{0}".format(xi_nm))

        print("样本X：\n{0}".format(X))
        print("猜中近邻矩阵X_nh为：\n{0}".format(X_nh))
        print("猜错近邻矩阵X_nm为：\n{0}".format(X_nm))

        print("开始计算属性统计量矩阵")
        # 计算X与猜中近邻矩阵，猜错近邻矩阵的统计分量贡献
        delta = self.calc_delta(X, X_nh, X_nm)
        return delta

    # 多分类分类问题
    def Relief_F(self):
        print("start Relief calculation")
        # self.dataset.datasplit(5000)
        X = self.dataset.getMinMaxDatasets()
        Y = self.dataset.getY()
        X_nh = np.zeros((X.shape))
        X_nm = {}
        print("开始计算样本X的猜中近邻矩阵和猜错近邻矩阵")
        for i in range(X.shape[0]):
            # 计算X的猜中近邻矩阵与猜错近邻矩阵
            print(i)
            xi_nh, xi_nm = dc.getNHM_F(X[i, :], Y[i], X, Y)
            X_nh[i, :] = xi_nh
            for label in xi_nm.keys():
                if label in X_nm:
                    X_nm[label] = np.vstack(X_nm[label], xi_nm[label]).copy()
                else:
                    X_nm[label] = (xi_nm[label]).copy()
            X_nm[i, :] = xi_nm
            print("猜中近邻：{0}".format(xi_nh))
            print("猜错近邻：{0}".format(xi_nm))

        print("样本X：\n{0}".format(X))
        print("猜中近邻矩阵X_nh为：\n{0}".format(X_nh))
        print("猜错近邻矩阵X_nm为：\n{0}".format(X_nm))

        print("开始计算属性统计量矩阵")
        # 计算X与猜中近邻矩阵，猜错近邻矩阵的统计分量贡献
        delta = self.calc_delta_F(X, X_nh, X_nm, Y)
        return delta


if __name__ == '__main__':
    data_file = './data/income_trainsets.csv'
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
    fs = FeatureSelection(data_file, data_transmit)
    delta = fs.Relief()
    print("采用Relief过滤式选择的属性统计量为：{0}".format(delta))
