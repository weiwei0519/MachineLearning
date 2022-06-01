# coding=UTF-8
# 回归（Regression）与最小二乘法（Least squares）
# 测试集数据准备

import numpy as np
import matplotlib.pyplot as plt
import pickle

# 创建训练样本
N = 1000
# CLASS 1
x_c1 = np.random.randn(N, 2)    # 从标准正态分布中返回一个或多个样本值, Nx2矩阵
x_c1 = np.add(x_c1, [10, 10])   # Nx2矩阵 + [10,10]
y_c1 = np.ones((N, 1), dtype=np.double)  # Nx1矩阵，[1.0]X1000
# CLASS 2
x_c2 = np.random.randn(N, 2)
x_c2 = np.add(x_c2, [2, 5])
y_c2 = np.zeros((N, 1), dtype=np.double)
# 扩展权向量
ex_c1 = np.concatenate((x_c1, np.ones((N, 1))), 1)
ex_c2 = np.concatenate((x_c2, np.ones((N, 1))), 1)
# 生成数据
data_x = np.concatenate((ex_c1, ex_c2), 0)  # ex_c1, ex_c2两个数组收尾拼接 (2*N)X3数组
data_y = np.concatenate((y_c1, y_c2), 0)

x1 = x_c1[:, 0].T    # [].T 数组转置
y1 = x_c1[:, 1].T
x2 = x_c2[:, 0].T
y2 = x_c2[:, 1].T

plt.plot(x1, y1, "bo", markersize=2)
plt.plot(x2, y2, "r*", markersize=2)
plt.show()

pickle_file = open('data_x.pkl', 'wb')
pickle.dump(data_x, pickle_file)            # pickle提供了持久化功能，主要用于列表，字典，集合，类等持久化
pickle_file.close()

pickle_file2 = open('data_y.pkl', 'wb')
pickle.dump(data_y, pickle_file2)
pickle_file2.close()