# coding=UTF-8
# 线性判别分析（LDA：Linear Discriminant Analysis）

import numpy as np
import pickle


class LDA():
    def Train(self, X, Y):
        """X为训练数据集，y为训练label"""
        # 将X训练集，按照Y in [0,1]进行分类
        X1 = np.array([X[i] for i in range(X.shape[0]) if Y[i] == 0])
        X2 = np.array([X[i] for i in range(X.shape[0]) if Y[i] == 1])

        # 求中心点，均值向量
        mju1 = np.mean(X1, axis=0)  # mju1是ndrray类型
        mju2 = np.mean(X2, axis=0)

        # dot(a, b, out=None) 计算矩阵乘法
        cov1 = np.zeros((X.shape[1], X.shape[1]), dtype=np.double)
        cov2 = np.zeros((X.shape[1], X.shape[1]), dtype=np.double)
        for i in range(len(X1)):
            cov1 += np.dot((X1[i] - mju1).T, (X1[i] - mju1))
        for i in range(len(X2)):
            cov2 += np.dot((X2[i] - mju2).T, (X2[i] - mju2))
        Sw = cov1 + cov2

        # 计算w
        print(mju1.shape[1])
        w = np.dot(np.mat(Sw).I, (mju1 - mju2).reshape((mju1.shape[1], 1)))

        print("类内散度矩阵：")
        print(Sw)
        print("判别权重矩阵：")
        print(w)

        # 记录训练结果
        self.mju1 = mju1  # 第1类的分类中心
        self.cov1 = cov1
        self.mju2 = mju2  # 第1类的分类中心
        self.cov2 = cov2
        self.Sw = Sw  # 类内散度矩阵
        self.w = w  # 判别权重矩阵

    def Test(self, X, Y):
        """X为测试数据集，y为测试label"""

        # 分类结果
        Y_new = np.dot((X), self.w)

        # 计算fisher线性判别式
        nums = len(Y)
        c1 = np.dot((self.mju1 - self.mju2).reshape(1, self.mju1.shape[1]), np.mat(self.Sw).I)
        c2 = np.dot(c1, (self.mju1 + self.mju2).reshape((self.mju1.shape[1], 1)))
        c = 1 / 2 * c2  # 2个分类的中心
        h = Y_new - c

        # 测试分类结果
        Y_test = []
        for i in range(nums):
            if h[i] >= 0:
                Y_test.append(0)
            else:
                Y_test.append(1)

        return Y_test


if '__main__' == __name__:
    # 产生分类数据
    pickle_file_X = open('./data/trainsets_X.pkl', 'rb+')
    pickle_file_Y = open('./data/trainsets_Y.pkl', 'rb+')
    X = pickle.load(pickle_file_X)
    Y = pickle.load(pickle_file_Y)
    pickle_file_X.close()
    pickle_file_Y.close()

    print("train datasets X is maxtrix {0} x {1}".format(X.shape[0], X.shape[1]))
    print(X)
    print("train datasets Y is maxtrix {0} x {1}".format(Y.shape[0], Y.shape[1]))
    print(Y)

    N = np.size(Y)

    # LDA线性判别分析(二分类)训练
    lda = LDA()
    lda.Train(X, Y)

    # model test, load testsets.
    pickle_file_test_X = open('./data/testsets_X.pkl', 'rb+')
    pickle_file_test_Y = open('./data/testsets_Y.pkl', 'rb+')
    test_X = pickle.load(pickle_file_test_X)
    test_Y = pickle.load(pickle_file_test_Y)
    pickle_file_test_X.close()
    pickle_file_test_Y.close()

    print("test datasets X is maxtrix {0} x {1}".format(test_X.shape[0], test_X.shape[1]))
    print(test_X)
    print("test datasets Y is maxtrix {0} x {1}".format(test_Y.shape[0], test_Y.shape[1]))
    print(test_Y)

    test_N = np.size(test_Y)

    predict_Y = lda.Test(test_X, test_Y)

    # 混淆矩阵
    TP = 0  # 真正例
    FN = 0  # 假反例
    FP = 0  # 假正例
    TN = 0  # 真反例

    # 通过后验概率，预测Y ~ （0,1）
    for i in range(test_N):
        if predict_Y[i] == 1:
            if test_Y[i] == predict_Y[i]:
                TP += 1
            else:
                FP += 1
        else:
            if test_Y[i] == predict_Y[i]:
                TN += 1
            else:
                FN += 1

    print("真正例TP = {0} / 假正例FP = {1} / 假反例FN = {2} / 真反例TN = {3}".format(TP, FP, FN, TN))
    # 计算查准率 P 和 查全率（召回率）R
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print("查准率P = {0} / 查全率R = {1}".format(P, R))
    F1 = (2 * P * R) / (P + R)
    print("F1 Score = {0}".format(F1))
