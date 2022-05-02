# coding=UTF-8
# 计算矩阵的奇异值

import numpy as np

a = np.random.randint(-10, 10, (4, 3)).astype(float)
print(a)
print("-----------------")
u, sigma, vT = np.linalg.svd(a)
print(u)
print("-----------------")
print(sigma)
print("-----------------")
print(vT)
print("-----------------")

# 将sigma 转成矩阵
SigmaMat = np.zeros((4, 3))
SigmaMat[:3, :3] = np.diag(sigma)
print(SigmaMat)
print("------验证-------")
a_ = np.dot(u, np.dot(SigmaMat, vT))
print(a_)
