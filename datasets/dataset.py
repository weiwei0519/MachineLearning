# coding=UTF-8
# 降维，西瓜书P225

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement


class DataSet:
    X = []  # X数据集
    Y = []  # Y数据集
    d = 0  # 数据集维度
    X_zs = []  # Z-Score标准化
    X_minmax = []  # Max-Min标准化
    X_maxabs = []  # MaxAbs标准化
    X_rob = []  # RobustScaler标准化
    X_central = []  # 数据中心化
    X_norm = []
    X_stand = []

    def __init__(self, file_name, data_transmit):
        print("Datasets initial")
        datasets = pd.read_csv(file_name, sep="\s*,\s*")  # sep="\s*,\s*"去掉空格
        datasets = datasets.replace(data_transmit)
        self.X = np.array(datasets.drop(columns='income'))
        self.Y = np.array(datasets['income'])
        self.d = self.X.shape[1]

    def getZScoreDatasets(self):
        # Z-Score标准化
        # 建立StandardScaler对象
        zscore = preprocessing.StandardScaler()
        # 标准化处理
        self.X_zs = zscore.fit_transform(self.X)
        return self.X_zs

    def getMinMaxDatasets(self):
        # Max-Min标准化
        # 建立MinMaxScaler对象
        minmax = preprocessing.MinMaxScaler()
        # 标准化处理
        self.X_minmax = minmax.fit_transform(self.X)
        return self.X_minmax

    def getMaxAbsDatasets(self):
        # MaxAbs标准化
        # 建立MinMaxScaler对象
        maxabs = preprocessing.MaxAbsScaler()
        # 标准化处理
        self.X_maxabs = maxabs.fit_transform(self.X)
        return self.X_maxabs

    def getRobustDatasets(self):
        # RobustScaler标准化
        # 建立RobustScaler对象
        robust = preprocessing.RobustScaler()
        # 标准化处理
        self.X_rob = robust.fit_transform(self.X)
        return self.X_rob

    def getCentralizedDatasets(self):
        X = self.X
        self.X_central = X - np.sum(X, axis=0) / X.shape[0]
        return self.X_central

    def getNormalizedDatasets(self, axis=-1, order=2):
        X = self.X
        """ Normalize the dataset X """
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1
        self.X_norm = X / np.expand_dims(l2, axis)
        return self.X_norm

    def getStandardizedDatasets(self):
        """ Standardize the dataset X """
        X = self.X
        X_std = X
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        for col in range(np.shape(X)[1]):
            if std[col]:
                X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
        # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        self.X_stand = X_std
        return self.X_stand

    @staticmethod
    def calcZScoreDatasets(X):
        # Z-Score标准化
        # 建立StandardScaler对象
        zscore = preprocessing.StandardScaler()
        # 标准化处理
        datasets_zs = zscore.fit_transform(X)
        return datasets_zs

    @staticmethod
    def calcMinMaxDatasets(X):
        # Max-Min标准化
        # 建立MinMaxScaler对象
        minmax = preprocessing.MinMaxScaler()
        # 标准化处理
        data_minmax = minmax.fit_transform(X)
        return data_minmax

    @staticmethod
    def calcMaxAbsDatasets(X):
        # MaxAbs标准化
        # 建立MinMaxScaler对象
        maxabs = preprocessing.MaxAbsScaler()
        # 标准化处理
        data_maxabs = maxabs.fit_transform(X)
        return data_maxabs

    @staticmethod
    def calcRobustDatas(X):
        # RobustScaler标准化
        # 建立RobustScaler对象
        robust = preprocessing.RobustScaler()
        # 标准化处理
        data_rob = robust.fit_transform(X)
        return data_rob

    @staticmethod
    def calcCentralizedDatasets(X):
        return X - np.sum(X, axis=0) / X.shape[0]

    @staticmethod
    def calcNormalizedDatasets(X, axis=-1, order=2):
        """ Normalize the dataset X """
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1
        return X / np.expand_dims(l2, axis)

    @staticmethod
    def calcStandardizedDatasets(X):
        """ Standardize the dataset X """
        X_std = X
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        for col in range(np.shape(X)[1]):
            if std[col]:
                X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
        # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        return X_std

    @staticmethod
    def polynomial_features(X, degree):
        n_samples, n_features = np.shape(X)

        def index_combinations():
            combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
            flat_combs = [item for sublist in combs for item in sublist]
            return flat_combs

        combinations = index_combinations()
        n_output_features = len(combinations)
        X_new = np.empty((n_samples, n_output_features))

        for i, index_combs in enumerate(combinations):
            X_new[:, i] = np.prod(X[:, index_combs], axis=1)

        return X_new

    # 不重复随机抽取m行数据集计算
    @staticmethod
    def doDatasplit(X, m):
        # 生成m个不重复的随机数
        index = np.random.choice(np.arange(X.shape[0]), size=m, replace=False)
        return X[index]

    # 不重复随机抽取m行数据集计算
    def datasplit(self, m):
        # 生成m个不重复的随机数
        index = np.random.choice(np.arange(self.X.shape[0]), size=m, replace=False)
        self.X = self.X[index]
        self.Y = self.Y[index]

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def datasets_show(self):
        # 可视化数据展示
        # 建立数据集列表
        data_list = [self.X, self.X_zs, self.X_minmax, self.X_maxabs, self.X_rob]
        # 创建颜色列表
        color_list = ['blue', 'red', 'green', 'black', 'pink']
        # 创建标题样式
        title_list = ['source data', 'zscore', 'minmax', 'maxabs', 'robust']

        # 设置画幅
        plt.figure(figsize=(9, 6))
        # 循环数据集和索引
        for i, dt in enumerate(data_list):
            # 子网格
            plt.subplot(2, 3, i + 1)
            # 数据画散点图
            plt.scatter(dt[:, 0], dt[:, 1], c=color_list[i])
            # 设置标题
            plt.title(title_list[i])
        # 图片储存
        plt.savefig('xx.png')
        # 图片展示
        plt.show()
