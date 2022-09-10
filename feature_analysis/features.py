# coding=UTF-8
# 特征选择的分析方法
# 

'''
@File: features.py
@Author: Wei Wei
@Time: 2022/9/8 21:42
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import pandas as pd
import pandas_profiling
from utils.pathutil import PathUtil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing

project_path = PathUtil()
# data = 'https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/automobile.csv'
data_path = project_path.rootPath + '/datasets/data/income_trainsets.csv'
data = pd.read_csv(data_path)
data.info()
data = data.sample(1000)
# 对非数值型变量进行预处理
f_names = ['workclass', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
for x in f_names:
    label = preprocessing.LabelEncoder()
    data[x] = label.fit_transform(data[x])

# pfr = pandas_profiling.ProfileReport(data)
# pfr.to_file("./profile_report.html")

# 相关性分析
corr_data = data.corr()
(corr_data.loc['income'].plot(kind='barh', figsize=(12, 10)))
# pd.plotting.scatter_matrix(data,
#                            figsize=(14, 14),
#                            c='k',
#                            marker='o',
#                            diagonal='hist',
#                            alpha=0.8,
#                            range_padding=0.1)
#
#
# f, ax = plt.subplots(figsize=(14, 9))
# sns.heatmap(corr_data, vmax=0.8, square=True)
# cols = corr_data.nlargest(14, 'income')['income'].index
# cm = np.corrcoef(data[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm,
#                  cbar=True,
#                  annot=True,
#                  square=True,
#                  fmt='.2f',
#                  annot_kws={'size': 10},
#                  yticklabels=cols.values,
#                  xticklabels=cols.values)
# plt.show()
