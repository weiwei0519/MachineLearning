# coding=UTF-8
# 非负矩阵分解算法（Non-negative Matrix Factorizations）

# NMF算法：非负矩阵分解算法。
# **目标：**将一个大矩阵分解成两个稍小的矩阵（利用矩阵的乘法）。
# 要求：待分解矩阵不能有负值。因为负值对于数据是无效的。
#
# 方法：
# 假定有一个元数据矩阵X，目标是将其分解成两个非负矩阵W和H相乘的形式。
# ** V = W * H ** （这边需要注意一些维度也就是角标，我就会直接写了）
# 其中，W称为权重系数矩阵，而H则为特征向量（可以反过来说都没关系，只是个符号表示）

import numpy as np

steps = 10000  # 最大迭代步数
alpha = 0.0001  # 学习率
beta = 0.02  # 控制特征向量，避免出现非常大的值


def matrix_factorisation(X, W, H, K, steps=5000, alpha=0.0002, beta=0.02):
    H = H.T
    for st in range(steps):
        # 这里注意为何要双循环，实际上对每个R中的值都要求误差
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i, j] > 0:  # 保证R没有负值
                    # numpy dot计算点积
                    eij = X[i, j] - np.dot(W[i, :], H[:, j])
                    for k in range(K):
                        W[i, k] = W[i, k] + alpha * (2 * eij * H[k, j] - beta * W[i, k])
                        H[k, j] = H[k, j] + alpha * (2 * eij * W[i, k] - beta * H[k, j])
        eR = np.dot(W, H)  # P*Q的实际值
        e = 0
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i, j] > 0:
                    e = e + pow(X[i, j] - eR[i, j], 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(W[i, k], 2) + pow(H[k, j], 2))
        if e < 0.001:
            break
    return W, H.T


X = [[5, 3, 1, 1], [4, 2, 3, 1], [1, 1, 3, 5], [1, 2, 5, 4], [7, 1, 5, 4]]
X = np.array(X)

N = len(X)
M = len(X[0])
print("X is a %d x %d matrix." % (N, M))
print(X)

K = 2

W = np.random.rand(N, K)
H = np.random.rand(M, K)

nW, nH = matrix_factorisation(X, W, H, K)
nX = np.dot(nW, nH.T)
print("W matrix is: ")
print(nW)
print("H matrix is: ")
print(nH)
print("nX matrix is: ")
print(nX)
