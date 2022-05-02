# coding=UTF-8
# 计算矩阵的奇异值

import numpy as np
import pandas as pd
from scipy.io import loadmat

# 读取数据，使用自己数据集的路径。
train_data_mat = loadmat("../data/train_data2.mat")
train_data = train_data_mat["Data"]
print(train_data.shape)

## 特征值分解
# 数据必需先转为浮点型，否则在计算的过程中会溢出，导致结果不准确
train_dataFloat = train_data / 255.0
# 计算特征值和特征向量
eval_sigma1,evec_u = np.linalg.eigh(train_dataFloat.dot(train_dataFloat.T))

## 计算右奇异矩阵
#降序排列后，逆序输出
eval1_sort_idx = np.argsort(eval_sigma1)[::-1]
# 将特征值对应的特征向量也对应排好序
eval_sigma1 = np.sort(eval_sigma1)[::-1]
evec_u = evec_u[:,eval1_sort_idx]
# 计算奇异值矩阵的逆
eval_sigma1 = np.sqrt(eval_sigma1)
eval_sigma1_inv = np.linalg.inv(np.diag(eval_sigma1))
# 计算右奇异矩阵
evec_part_v = eval_sigma1_inv.dot((evec_u.T).dot(train_dataFloat))