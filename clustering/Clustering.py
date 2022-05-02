# coding=UTF-8
# 聚类，西瓜书P200

import numpy as np
from collections import Counter
import math
import pandas as pd
import datetime


class Clustering:
    D = []  # X数据集
    Y = []  # Y数据集
    k = 2  # 分类簇k的数值
    p = 2  # 闵可夫斯基距离计算的指数
    categorical = False

    def __init__(self, file_name, data_transmit, k=2, p=2):
        print("Clustering initial")
        datasets = pd.read_csv(file_name, sep="\s*,\s*")  # sep="\s*,\s*"去掉空格
        datasets = datasets.replace(data_transmit)
        self.D = np.array(datasets.drop(columns='income'))
        self.Y = np.array(datasets['income'])
        self.k = k
        self.p = p

        # 分析数据集是连续属性，还是存在离散属性
        D = self.D
        m, n = D.shape
        categories = {}
        for l in range(n):
            if sum(type(value) in (int, float) for value in np.ravel(D[:, l])) < m:
                categories[l] = 'categorical_attribute'
            else:
                categories[l] = 'continuous_attribute'

        if 'categorical_attribute' in categories.values():
            self.categorical = True

    # 获取数据集
    def getDatasets(self):
        return self.D

    # 不重复随机抽取m行数据集计算
    def datasplit(self, m):
        # 生成m个不重复的随机数
        index = np.random.choice(np.arange(self.D.shape[0]), size=m, replace=False)
        return self.D[index]

    # 计算样本xi~xj的“闵可夫斯基距离”
    def calc_mk_dist(self, xi, xj, p, w):
        # xi, xj 入参为numpy数组
        if xi.shape != xj.shape:
            print("样本xi，xj不是同一维度")
            return
        dist_mk = 0.0
        z = w.reshape(-1, 1)
        # dist_mk = np.ravel(np.power(np.dot((np.power(xi - xj, p)).reshape(1, -1), z), 1 / p))
        dist_mk = np.power(np.dot(np.power(xi - xj, p), z), 1 / p)
        return dist_mk[0]

    # 计算样本xi~xj的“欧氏距离”
    def calc_ed_dist(self, xi, xj, w):
        dist_ed = 0.0
        # 欧氏距离为：p=2时的闵可夫斯基距离
        dist_ed = self.calc_mk_dist(xi, xj, 2, w)
        return dist_ed

    # 计算样本xi~xj的“曼哈顿距离”
    def calc_man_dist(self, xi, xj, w):
        dist_man = 0.0
        # 曼哈顿距离为：p=1时的闵可夫斯基距离
        dist_man = self.calc_mk_dist(xi, xj, 1, w)
        return dist_man

    # 计算样本xi~样本集合Cj的“闵可夫斯基距离”,返回是一个距离数组
    def calc_mk_dist_mat(self, xi, Cj, p, w):
        # xi, xj 入参为numpy数组
        if xi.shape[0] != Cj.shape[1]:
            print("样本xi，xj不是同一维度")
            return
        dist_mk = 0.0
        z = w.reshape(-1, 1)
        # dist_mk = np.ravel(np.power(np.dot((np.power(xi - xj, p)).reshape(1, -1), z), 1 / p))
        dist_mk = np.power(np.dot(np.power(xi - Cj, p), z), 1 / p)
        return np.ravel(dist_mk)

    # 计算样本xi~xj的“欧氏距离”
    def calc_ed_dist_mat(self, xi, xj, w):
        dist_ed = 0.0
        # 欧氏距离为：p=2时的闵可夫斯基距离
        dist_ed = self.calc_mk_dist_mat(xi, xj, 2, w)
        return dist_ed

    # 计算样本xi~xj的“曼哈顿距离”
    def calc_man_dist_mat(self, xi, xj, w):
        dist_man = 0.0
        # 曼哈顿距离为：p=1时的闵可夫斯基距离
        dist_man = self.calc_mk_dist_mat(xi, xj, 1, w)
        return dist_man

    # 计算属性u上两个离散值a, b的“VDM”距离
    def calc_VDM(self, Du, a, b, p):
        # Du 的格式为：m x 2，第一列为属性u列，第二列为簇的分类列
        VDMp = 0.0
        m_u = dict(Counter(np.ravel(Du[:, 0]).tolist()))  # 属性列的属性值分布统计
        m_uxi = m_u[a]
        m_uxj = m_u[b]
        k_u = dict(Counter(np.ravel(Du[:, 1]).tolist()))  # 分类簇的分布统计
        kvalues_u = k_u.keys()
        for kvalue in kvalues_u:
            indexlist = np.where(Du[:, 1] == kvalue)
            k = indexlist[0]
            m_k = dict(Counter(np.ravel(Du[k, 0]).tolist()))  # 属性列中，在k簇下的属性值的分布统计
            try:
                m_kxi = m_k[a]
            except KeyError:
                m_kxi = 0  # 在这个簇没有xi属性值时，赋值0

            try:
                m_kxj = m_k[b]
            except KeyError:
                m_kxj = 0  # 在这个簇没有xj属性值时，赋值0

            VDMp += np.power(math.fabs(m_kxi / m_uxi - m_kxj / m_uxi), p)

        return VDMp

    # 对xi，xj的混合属性，利用“闵可夫斯基距离”和“VDM”距离综合计算xi~xj的距离
    # w为权值数组，1 x n，采用加权计算
    def calc_dist_comb_weight(self, D, xi, xj, p, w):
        m, n = D.shape
        u = D[:, -1]  # 分类簇列 暂时无法使用，有问题
        if len(w) != n:
            print("权重维度与数据集维度不一致")
            return
        if xi.shape[0] != n or xi.shape[0] != xj.shape[0]:
            print("属性维度不一致")
            return
        dist_wmk = 0.0

        if not self.categorical:
            # 对连续属性，计算闵可夫斯基距离
            dist_wmk += self.calc_mk_dist(xi, xj, p, w)
        else:
            # 如果存在离散属性列，需要计算VDM，暂未实现
            return

        dist_wmk = np.power(dist_wmk, 1 / p)

        return dist_wmk

    # 对xi，xj的混合属性，利用“闵可夫斯基距离”和“VDM”距离综合计算xi~xj的距离，权重矩阵为1/n
    def calc_dist_comb(self, D, xi, xj, p):
        m, n = D.shape
        w = np.ones((n)) / n
        # 权重为1/n时，加权算法退化为MindovDM
        MindovDMp = self.calc_dist_comb_weight(D, xi, xj, p, w)

        return MindovDMp

    # 计算两个样本集合X与Z之间的距离：豪斯多夫距离 P220
    def calc_Hausdorff_dist(self, C_X, C_Z):
        # 公式：dist_H(X, Z) = max(dist_h(X, Z), dist_h(Z, X))
        x_m, x_n = C_X.shape
        z_m, z_n = C_Z.shape
        w = np.ones((x_n)) / x_n
        max_dist_1 = 0.0
        max_dist_2 = 0.0
        min_dist = float('inf')
        # dist_h(X, Z) = max(min(||x-z||2))
        for xi in range(x_m):
            # 少一层循环,提高性能
            dist_list = self.calc_ed_dist_mat(C_X[xi, :], C_Z, w)
            min_i = np.min(dist_list)
            if min_i < min_dist:
                min_dist = min_i

            max_dist_1 = max(max_dist_1, min_dist)

        min_dist = float('inf')
        # dist_h(Z, X) = max(min(||z-x||2))
        for zi in range(z_m):
            # 少一层循环,提高性能
            dist_list = self.calc_ed_dist_mat(C_Z[zi, :], C_X, w)
            min_i = np.min(dist_list)
            if min_i < min_dist:
                min_dist = min_i

            max_dist_2 = max(max_dist_2, min_dist)

        return max(max_dist_1, max_dist_2)

    # 计算两个样本集合X与Z之间的平均距离
    def calc_avg_dist(self, C_X, C_Z):
        # 公式：dist_H(X, Z) = max(dist_h(X, Z), dist_h(Z, X))
        x_m, x_n = C_X.shape
        z_m, z_n = C_Z.shape
        w = np.ones((x_n)) / x_n
        dist = 0.0
        # dist_h(X, Z) = max(min(||x-z||2))
        for xi in range(x_m):
            # 少一层循环,提高性能
            dist_list = self.calc_ed_dist_mat(C_X[xi, :], C_Z, w)
            dist += np.sum(dist_list)

        return dist / (x_m * z_m)

    # 计算样本均值
    def calc_dataset_avg(self, D):
        m, n = D.shape
        sumX = np.sum(D, axis=0)  # 二维数组，按照axis指定的维度求和
        return sumX / m

    # k均值聚类算法实现
    def k_clustering(self):
        m, n = self.D.shape
        D = self.D
        k = self.k
        p = self.p  # 默认采用欧式距离计算
        # step1: 从D中随机选择k个样本作为初始均值向量{miu_1, miu_2, ... miu_k}
        miu_k = np.random.randint(0, m, k)
        miu = D[miu_k, :]
        has_update = True
        round = 1
        while has_update:
            C = {}  # 聚簇数据集，每行聚簇类，存数据集的下标
            # step2: 计算样本xj与各均值向量miu_i的距离
            for j in range(m):
                min_dji = float('inf')
                gamma_j = 0
                for i in range(k):
                    dji = self.calc_dist_comb(D, D[j, :], miu[i, :], p)
                    if min_dji > dji:
                        min_dji = dji
                        gamma_j = i
                if gamma_j in C:
                    C[gamma_j].append(j)
                else:
                    C[gamma_j] = [j]

            # step3: 在新的聚簇分类C下，计算新均值向量
            has_update = False
            for i in range(k):
                D_Ck = np.array([D[j, :] for j in C[i]])
                avg_Ck = self.calc_dataset_avg(D_Ck)
                if (miu[i, :] != avg_Ck).all():
                    miu[i, :] = avg_Ck
                    has_update = True
            # 直到没有更新，推出循环
            print("第{0}轮迭代计算出{1}个分类簇的均值样本为：".format(round, k))
            print(miu)
            for Ck, value in C:
                print("第{0}分类簇的样本数是：{1}".format(Ck, len(value)))
            round += 1
        return C, miu

    # LVQ学习向量量化算法实现，eta为学习率
    def LVQ(self, eta=0.5):
        m, n = self.D.shape
        X = self.D
        Y = self.Y
        p = self.p  # 默认采用欧式距离计算
        k = self.k
        labels = np.unique(np.ravel(Y)).shape[0]  # 现有Y的分类数

        # step1: 从D中随机选择k个样本作为初始原型向量{p_1, p_2, ... p_k}
        # LVQ原型向量的个数，选取Y分类的label个数，不同分类随机选择一个作为初始化原型向量P
        not_ok = True
        while not_ok:
            P_k = np.random.randint(0, m, k)
            P = X[P_k, :]
            y_rand = Y[P_k]
            chosen_labels = np.unique(np.ravel(y_rand))
            if chosen_labels.shape[0] == labels:
                not_ok = False
        max_round = 20
        while round in range(1, max_round + 1):
            # step2: 计算样本xj与各原型向量p_i的距离
            j_list = np.random.randint(0, m, m)
            for j in j_list:
                min_dji = float('inf')
                min_i = 0
                for i in range(k):
                    dji = self.calc_dist_comb(X, X[j, :], P[i, :], p)
                    if min_dji > dji:
                        min_dji = dji
                        min_i = i
                if (Y[j] == Y[P_k[min_i]]).all():
                    P[min_i, :] = P[min_i, :] + (X[j, :] - P[min_i, :]) * eta
                else:
                    P[min_i, :] = P[min_i, :] - (X[j, :] - P[min_i, :]) * eta

            print("第{0}轮迭代计算出{1}个分类簇的均值样本为：".format(round, k))
            print(P)

        return P

    # 层次聚类 P214
    def hierarchical_clustering(self):
        D = self.datasplit(5000)
        m, n = D.shape
        k = self.k
        C = {}  # 聚类簇
        for i in range(m):
            C[i] = np.array([D[i, :]])  # 初始化聚类簇，每个样本归类于一簇

        print("初始聚类簇数为:{0}".format(len(C)))
        # 初始计算聚类簇距离矩阵
        print("开始计算距离矩阵")
        M = np.identity(m, dtype=float) * float('inf')  # 创建对角矩阵,对角线预算设置很大,这样后续寻找矩阵极小值时比较容易
        for i in range(m):
            for j in range(i + 1, m):
                M[i, j] = self.calc_Hausdorff_dist(C[i], C[j])  # 豪斯多夫距离计算复杂度较大
                # M[i, j] = self.calc_avg_dist(C[i], C[j])  # 计算平均距离
                M[j, i] = M[i, j]

        print("距离矩阵M = {0} x {1}".format(M.shape[0], M.shape[1]))

        q = m  # 初始化聚类族个数
        while q > k:
            # 在距离矩阵中，找出距离最近的两个聚类簇的下标，有可能是多个
            index = np.array(np.where((M == np.min(M))))
            print("当前处理的index为：{0}".format(index))
            index = index.T[np.lexsort(index[::-1, :])].T  # 查询到距离最近的下标二维矩阵，按照第一行排序
            # 因为距离最近的下标是成对出现的，所以只需要处理一半就够了
            index = index[:, 0:int(index.shape[1] / 2)]
            index_del = []
            # 将距离最近的聚类簇合并，可能同时存在多个最小值
            for l in range(index.shape[1]):
                min_i = index[0, l]
                min_j = index[1, l]
                if min_i in index_del or min_j in index_del:
                    continue  # 说明已经处理过这个分类簇了，跳出本轮循环。
                C[min_i] = np.vstack((C[min_i], C[min_j]))
                # 重算距离矩阵中第min_i行的距离,并更新距离矩阵
                for j in range(m):
                    if j != min_i and M[min_i, j] != float('inf'):
                        M[min_i, j] = self.calc_Hausdorff_dist(C[min_i], C[j])

                # 重算距离矩阵中第min_i列的距离,并更新距离矩阵
                for i in range(m):
                    if i != min_i and M[i, min_i] != float('inf'):
                        M[i, min_i] = self.calc_Hausdorff_dist(C[i], C[min_i])

                del C[min_j]  # 删除min_j分类簇
                index_del.append(min_j)
                # 将min_j分类簇所对应的距离矩阵中的第min_j行和min_j列都赋值为极大值(float('inf')),后续找距离矩阵极小值时,不会再触碰到了
                M[min_j, :] = float('inf')
                M[:, min_j] = float('inf')

            print(len(C))
            q = len(C)

        counts = 0
        for key, value in C.items():
            counts += value.shape[0]
        print("已分类数据集数为:{0}".format(counts))

        return C


if __name__ == '__main__':
    data_file = './data/income_trainsets.csv'
    k = 20
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
        "native-country": {"United-States": 1, "Cambodia": 2, "England": 3, "Puerto-Rico": 4, "Canada": 5,
                           "Germany": 6,
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
    current_time = datetime.datetime.now()
    print("开始执行:{0}".format(current_time))
    clu = Clustering(data_file, data_transmit, k)
    print(clu.getDatasets()[0:10, :])  # 输出前10行看看对不对
    # C, miu = clu.k_clustering()  # k均值
    # print("C: ")
    # print(C)
    # print("miu: ")
    # print(miu)

    # P = clu.LVQ()  # LVQ聚类
    # print("原型向量P: ")
    # print(P)

    C = clu.hierarchical_clustering()  # 层次聚类
    print("C: ")
    for key, value in C.items():
        print("分类{0}:".format(key))
        print(value)

    current_time = datetime.datetime.now()
    print("结束执行:{0}".format(current_time))
