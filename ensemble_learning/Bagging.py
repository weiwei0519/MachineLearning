# coding=UTF-8
# 集成学习 Bagging算法 + 随机森林选择属性的方式实现，西瓜书图8.3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from collections import Counter
import threading
from threading import Lock, Thread
import math

np.set_printoptions(suppress=True)  # suppress=True 取消科学记数法

# 设置出图显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据集采用adult_income，并且是经过量化的。
headers = ['age', 'workclass', 'fnlwgt', 'education', 'education_years', 'marital', 'occupation',
           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_perweek', 'native_country',
           'income']


# collections.Counter方法存在丢数据的情况，所以自己又写了一个
def DatasetsCounter(D):
    counter = {}
    m, n = D.shape
    for col in range(n):
        counter[col] = dict(Counter(D[:, col]))
    return counter


##### 计算信息熵 ######
def calcEntropy(D):
    numEntries = len(D)  # 样本数
    labelCounts = {}  # 创建一个数据字典：key是目标分类的类别，value是属于该类别的样本个数
    for featVec in D:  # 遍历整个数据集，每次取一行
        currentLabel = featVec[-1]  # 取该行最后一列的值
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    entropy = 0.0  # 初始化信息熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        entropy -= prob * np.log2(prob)  # log base 2  计算信息熵，越小说明数据集分类约属于同一类 西瓜书P75 (4.1)
        # entropy -= prob * math.log(prob, 2)
    return entropy


##### 按给定的特征划分数据 #########
def splitDataSet(D, col, value):
    indexlist = np.where(D[:, col] == value)
    row = indexlist[0]
    Dv = np.hstack((D[row, :col], D[row, col + 1:]))
    return Dv


# 对连续列，进行分段平均，用每段平均来计算信息增益，减少运算量
def split_features(list, step):
    newlist = []
    alist = []
    i = 0
    col = 0
    for i in range(len(list)):
        list_2 = list[i:i + step]
        i += step
        the_sum = sum(list_2)
        the_length = len(list_2)
        the_average = round(the_sum / the_length)
        alist.append(the_average)
        col += 1
        if col >= 100:
            newlist.append(alist)
            col = 0
            alist = []
    return newlist


##### 计算信息增益 ######
## D 数据集，feature_values属性列的值列表，feature_col属性列的列号
def calcGain(D, feature_values, feature_col):
    ent_D = calcEntropy(D)  # 计算入参数据集的信息熵
    sum_ent_Dv = 0.0
    # Gain(D, a) = Ent(D) - sum(|Dv|/|D| * Ent(Dv))   Dv为属性a在v取值下的数据子集，|D|为数据集的数据个数
    split_value = 0
    min_ent_Dv = float('inf')  # 先设置一个很大的值
    ent_Dv_i = 0
    for v in feature_values:  # 计算每种划分方式的信息熵
        Dv = splitDataSet(D, feature_col, v)
        prob = Dv.shape[0] / D.shape[0]
        ent_Dv_i = prob * calcEntropy(Dv)
        if min_ent_Dv > ent_Dv_i:
            # 西瓜书P75页，Ent(D)的值越大（书中时）
            split_value = v
            min_ent_Dv = ent_Dv_i
        sum_ent_Dv += ent_Dv_i

    Gain_D_a = ent_D - sum_ent_Dv
    # print("Gain(D<{0}x{1}>, {2}) = {3}".format(len(D), len(D[0]), a_label, Gain_D_a))

    return Gain_D_a, split_value


min_ent_Dv = float('inf')  # 通过全局变量在多线程之间共享
sum_ent_Dv = 0.0
split_value = 0.0


##### 计算信息增益 ######
## 对于连续列，计算Gain比较消耗性能，采用多线程的方式计算，此方法为子线程
def calcGain_run(D, feature_values, feature_col, lock):
    # Gain(D, a) = Ent(D) - sum(|Dv|/|D| * Ent(Dv))   Dv为属性a在v取值下的数据子集，|D|为数据集的数据个数
    t_split_value = 0
    t_sum_ent_Dv = 0.0
    t_min_ent_Dv = float('inf')  # 先设置一个很大的值
    ent_Dv_i = 0
    for v in feature_values:  # 计算每种划分方式的信息熵
        Dv = splitDataSet(D, feature_col, v)
        prob = Dv.shape[0] / D.shape[0]
        ent_Dv_i = prob * calcEntropy(Dv)
        if t_min_ent_Dv > ent_Dv_i:
            # 西瓜书P75页，Ent(D)的值越大（书中时）
            t_split_value = v
            t_min_ent_Dv = ent_Dv_i
        t_sum_ent_Dv += ent_Dv_i

    # 定义线程安全的局部变量
    global sum_ent_Dv, min_ent_Dv, split_value
    lock.acquire()
    sum_ent_Dv += t_sum_ent_Dv
    if min_ent_Dv > t_min_ent_Dv:
        split_value = t_split_value
        min_ent_Dv = t_min_ent_Dv
    lock.release()


##### 计算信息增益 ######
## 对于连续列，计算Gain比较消耗性能，采用多线程的方式计算，此方法为主线程
def calcGain_threads(D, feature_values, feature_col):
    ent_D = calcEntropy(D)  # 计算入参数据集的信息熵
    # 定义线程安全的局部变量
    global sum_ent_Dv, min_ent_Dv, split_value
    sum_ent_Dv = 0.0
    # Gain(D, a) = Ent(D) - sum(|Dv|/|D| * Ent(Dv))   Dv为属性a在v取值下的数据子集，|D|为数据集的数据个数
    split_value = 0
    min_ent_Dv = float('inf')  # 先设置一个很大的值
    ent_Dv_i = 0
    split_featurelists = split_features(feature_values, 10)
    threads = []
    for fList in split_featurelists:  # 计算每种划分方式的信息熵
        lock = Lock()
        threads.append(threading.Thread(target=calcGain_run, args=(D, fList, feature_col, lock,)))

    print("第{0}列的候选分析划分点有{1}个，拆分为{2}个线程执行".format(feature_col, len(feature_values), len(threads)))

    for thread in threads:
        thread.setDaemon(True)  # 把子线程设置为守护线程，必须在start()之前设置
        thread.start()  # 启动线程

    for thread in threads:
        thread.join()  # 设置主线程等待子线程结束

    Gain_D_a = ent_D - sum_ent_Dv

    return Gain_D_a, split_value


##### 基学习器——决策树桩 选取当前数据集下，用于划分数据集的最优特征，并基于最优特征，计算划分概率
def calc_bestfeature_prob(X, Y, counter):
    D = np.hstack((X, Y))
    numFeatures = X.shape[1]  # 获取当前数据集的特征个数
    best_Feature_col = -1  # 最优的特征列号
    best_Feature_name = ""  # 最优的特征分类值
    max_Gain_D = -1.0  # 最优信息增益，信息增益最小值为0
    best_split_value = 0
    best_feat_discreted = False

    # 采用随机森林的方法，随机选择k个属性进行分类
    k = int(np.round(np.log2(numFeatures)))
    feat_candidate = sorted(random.sample(list(range(numFeatures)), k))
    # feat_candidate = np.random.randint(0, numFeatures, size=k)
    # feat_candidate = np.unique(feat_candidate)
    print("随机森林法随机选择的属性列有：{0}".format(feat_candidate))

    for i in feat_candidate:
        featList = np.ravel(X[:, i]).tolist()  # 获取当前属性列的所有数据，转换为list类型
        feature_group = dict(Counter(featList))  # 使用Counter函数计算这一列的各特征数量
        feature_values = sorted(feature_group.keys())  # 获取当前特征值
        # 判断当前属性列是离散还是连续
        discreted_fea = False
        if len(feature_values) < D.shape[0] * 0.00001:
            discreted_fea = True

        Gain_D_a, split_value = calcGain(D, feature_values, i)
        # if discreted_fea:
        #     Gain_D_a, split_value = calcGain(D, feature_values, i)
        # else:
        #     # 对于连续列，计算Gain采用多线程
        #     Gain_D_a, split_value = calcGain_threads(D, feature_values, i)

        # 西瓜书P75页，信息增益越大，用当前属性feature来划分所获得的“纯度提升”越大
        if (Gain_D_a > max_Gain_D):
            max_Gain_D = Gain_D_a
            best_split_value = split_value
            best_Feature_name = headers[i]
            best_Feature_col = i
            best_feat_discreted = discreted_fea

    # 计算最佳属性列的分类概率：
    best_feature_prob = {}
    if best_feat_discreted:
        # 如果是离散列，计算每个离散值的分类概率；
        # 注意此处遍历的是最佳分类属性列所有的数据，不是入参X，因为入参X是经过随机采用的，有可能会出现某些属性值完全没有被采样到。
        col_counter = counter[best_Feature_col]
        featList = list(X[:, best_Feature_col])  # 在采样样本中，获取最佳分类属性列的所有数据，转换为list类型
        # feature_group = dict(Counter(featList))  # 使用Counter函数计算这一列的各特征数量
        feature_values = col_counter.keys()  # 获取当前特征值
        # 格式为：{feature_value_1:{label_1:prob; label_2:prob; ...}; feature_value_2:{label_1:prob; label_2:prob; ...}}
        for v in feature_values:
            feature_prob = {}
            y_v = np.ravel(Y[featList == v]).tolist()
            if len(y_v) > 0:
                label_stat = dict(Counter(y_v))
            else:
                # 引入拉普拉斯平滑因子
                label_stat = dict(Counter(np.ravel(Y).tolist()))
                for key in label_stat.keys():
                    label_stat[key] = 1

            label_values = label_stat.keys()
            for label_value in label_values:
                prob = label_stat[label_value] / Y.shape[0]
                feature_prob[label_value] = prob

            best_feature_prob[v] = feature_prob.copy()
    else:
        # 如果是连续列，计算split_value左右的分类概率（更精确的，也可以分区间段计算分类概率，对于多分类任务更适用）
        # 格式为：{best_split_value:best_split_value; best_split_prob:
        #                                   {left:{label_1:prob; label_2:prob};
        #                                    right:{label_1:prob; label_2:prob}}}
        featList = X[:, best_Feature_col]  # 获取最佳分类属性列的所有数据
        best_split_prob = {}
        # left 分类概率统计
        y_left = np.ravel(Y[featList < best_split_value]).tolist()
        left_label_stat = dict(Counter(y_left))
        left_label_values = left_label_stat.keys()
        left_feature_value_p = {}
        for left_label_value in left_label_values:
            prob = left_label_stat[left_label_value] / Y.shape[0]
            left_feature_value_p[left_label_value] = prob
        best_split_prob['left'] = left_feature_value_p.copy()
        # right 分类概率统计
        y_right = np.ravel(Y[featList >= best_split_value]).tolist()
        right_label_stat = dict(Counter(y_right))
        right_label_values = right_label_stat.keys()
        right_feature_value_p = {}
        for right_label_value in right_label_values:
            prob = right_label_stat[right_label_value] / Y.shape[0]
            right_feature_value_p[right_label_value] = prob
        best_split_prob['right'] = right_feature_value_p.copy()

        best_feature_prob['best_split_value'] = best_split_value
        best_feature_prob['best_split_prob'] = best_split_prob.copy()

    best_feature_prob['discreted'] = best_feat_discreted

    # 定义基学习器的返回输出
    h = {}
    h['best_feature_name'] = best_Feature_name
    h['best_feature_col'] = best_Feature_col
    h['max_Gain_D'] = max_Gain_D
    h['best_feature_prob'] = best_feature_prob.copy()

    return h


# 在基学习器h下，预测y
def h_predict(X, h):
    print("h: {0}".format(h))
    m, features = X.shape
    y_pred = np.zeros((m, 1))
    best_feature_col = h['best_feature_col']
    best_feature_prob = h['best_feature_prob']
    discreted = best_feature_prob['discreted']
    featList = list(X[:, best_feature_col])  # 获取最佳分类属性列的所有数据，转换为list类型
    if discreted:
        for i in range(len(featList)):
            # 判断测试集中，是否存在此属性的选项，如果不存在，找最近的选项（数字类型）
            if featList[i] in best_feature_prob:
                label_prob = best_feature_prob[featList[i]]
            else:
                for n in range(100):
                    if str(int(featList[i]) + n) in best_feature_prob:
                        label_prob = best_feature_prob[str(int(featList[i]) + n)]
                        break
                    elif str(int(featList[i]) - n) in best_feature_prob:
                        label_prob = best_feature_prob[str(int(featList[i]) - n)]
                        break
            y_pred[i] = max(label_prob, key=label_prob.get)  # 获得字典dict中value的最大值所对应的键的方法
    else:
        best_split_value = best_feature_prob['best_split_value']
        best_split_prob = best_feature_prob['best_split_prob']
        location = ''
        for i in range(len(featList)):
            location = 'left' if featList[i] < best_split_value else 'right'
            label_prob = best_split_prob[location]
            y_pred[i] = max(label_prob, key=label_prob.get)

    return y_pred


# 在集成学习模型H下，预测y，采用多数投票法
def H_predict(X, H):
    m, features = X.shape
    y_pred = np.zeros((m, 1))
    y_H = np.zeros((m, 1))
    for h in H:
        if np.sum(y_H) == 0:
            y_H = h_predict(X, h)
        else:
            y_H = np.hstack((y_H, h_predict(X, h)))

    # 采用多数投票法进行选择
    for i in range(m):
        y_row = dict(Counter(np.ravel(y_H[i, :]).tolist()))
        y_pred[i, :] = max(y_row.keys(), key=(lambda x: y_row[x]))
    return y_pred


def bagging_model_train(X, Y):
    m = X.shape[0]
    T = X.shape[1]  # 训练轮次，有大概率可以命中每个属性列
    H = []  # 决策树序列，格式{'best_feature_name':xx; 'best_feature_col':xx; 'max_Gain_D':xx;}
    error = []
    counter = DatasetsCounter(X)  # 随机采用，会导致有些属性值在基训练前中无数据，所以需要将数据集各个属性的所有候选值预先统计好
    for t in range(T):
        bootstrap_sampling = np.random.randint(0, m, m)
        X_bs = X[bootstrap_sampling]  # 自助采样
        Y_bs = Y[bootstrap_sampling]  # 自助采样
        # h = decision_stumps_MaxInfoGain(X_bs, Y_bs)  # 训练基学习器
        h = calc_bestfeature_prob(X_bs, Y_bs, counter)
        # if h in H: break
        H.append(h)
        print("第{0}轮基学习器h：{1}".format(t, h))
        # 计算并存储训练误差
        y_pred = h_predict(X, h)
        error.append(np.sum((Y - y_pred) * (Y - y_pred)) / m)  # 分类错误的均方差
    H = np.array(H)
    return H, error


# 多线程执行InfoGain计算
def run(x, Y, ts, Gains, lock):
    m = len(x)
    thread_Gains = []  # 存储当前线程的各个划分点下的信息增益
    print(f"thread_ts = {len(ts)}")
    for t in ts:
        Gain = 0
        Y_left = Y[x <= t]  # 左分支样本的标记
        Dl = len(Y_left)
        p_plus = sum(Y_left == 1) / Dl  # 左分支正例样本比例
        p_minus = sum(Y_left == -1) / Dl  # 左分支负例样本比例
        Gain += Dl / m * (np.log2(p_plus ** p_plus) + np.log2(p_minus ** p_minus))

        Y_right = Y[x > t]  # 右分支样本标记
        Dr = len(Y_right)
        p_plus = sum(Y_right == 1) / Dr  # 右分支正例样本比例
        p_minus = sum(Y_right == -1) / Dr  # 右分支负例样本比例
        Gain += Dr / m * (np.log2(p_plus ** p_plus) + np.log2(p_minus ** p_minus))
        thread_Gains.append(Gain)

    print(f"thread_Gains = {len(thread_Gains)}")
    # Gains为线程安全的局部变量
    lock.acquire()
    for Gain in thread_Gains:
        Gains.append(Gain)
    lock.release()


def decision_stumps_MaxInfoGain(X, Y):
    # 基学习器——决策树桩
    # 以信息增益最大来选择划分属性和划分点
    m, n = X.shape
    results = []  # 存储各个特征下的最佳划分点，左分支取值，右分支取值，信息增益
    # 采用随机森林的方法，随机选择k个属性进行分类
    k = int(np.round(np.log2(n)))
    attr_index = np.random.randint(0, n, size=k)
    attr_index = np.unique(attr_index)
    print("随机森林法随机选择的属性列有：{0}".format(attr_index))
    for i in attr_index:
        # i = 9
        x = X[:, i]  # i列特征取值
        x_values = np.unique(x)  # 去重并排序
        ts = (x_values[1:] + x_values[:-1]) / 2  # 候选划分点，相邻两个值的均值序列。
        # 当候选划分点比较多时，需要缩减循环次数，考虑增加步长。
        if len(ts) >= 1000:
            step = len(ts) // 1000
            if step == 1: step += 1
            index = list(filter(lambda l: l % step == 0, range(len(ts))))
            ts = [ts[i] for i in index]
        # print("第{0}列的候选分析划分点有{1}".format(i, len(ts)))
        # 当候选点比较多时，采用多线程
        Gains = []  # 存储各个划分点下的信息增益
        tns = []
        if len(ts) > 100:
            tns.append(ts[:100])
        else:
            tns.append(ts[:])
        p = 1
        while (p + 1) * 100 < len(ts):
            tns.append(ts[p * 100:(p + 1) * 100])
            p += 1
        if len(ts) > p * 100:
            tns.append(ts[p * 100:])

        threads = []
        for tn in tns:
            lock = Lock()
            threads.append(threading.Thread(target=run, args=(x, Y, tn, Gains, lock,)))

        print("第{0}列的候选分析划分点有{1}个，拆分为{2}个线程执行".format(i, len(ts), len(threads)))

        for thread in threads:
            thread.setDaemon(True)  # 把子线程设置为守护线程，必须在start()之前设置
            thread.start()  # 启动线程

        for thread in threads:
            thread.join()  # 设置主线程等待子线程结束
        print(f"Gains num = {len(Gains)}")
        best_t = ts[np.argmax(Gains)]  # 当前特征列的最佳划分点 argmax返回最大值的index
        best_gain = np.max(Gains)  # 当前特征列的最佳信息增益
        left_value = (np.sum(Y[x <= best_t]) >= 0) * 2 - 1  # 左分支取值(多数类的类别)
        right_value = (np.sum(Y[x > best_t]) >= 0) * 2 - 1  # 右分支取值（多数类的类别）
        results.append([best_t, left_value, right_value, best_gain])

    results = np.array(results)
    df = np.argmax(results[:, -1])  # df表示divide_feature, 划分特征
    h = [df] + list(results[df, :3])  # 划分特征，划分点，左枝取值，右枝取值
    return h


def predict(H, X):
    # 预测结果用平均法集成
    pre = np.zeros((X.shape[0], 1))
    for h in H:
        df, t, lv, rv = h  # 划分特征列，划分点，左枝取值，右枝取值
        pre += ((X[:, int(df)] <= t) * lv + (X[:, int(df)] > t) * rv).reshape(-1, 1)
    return np.sign(pre)


if __name__ == '__main__':
    pickle_file_X = open('./data/trainsets_X.pkl', 'rb+')
    pickle_file_Y = open('./data/trainsets_Y.pkl', 'rb+')
    X_train = pickle.load(pickle_file_X)
    y_train = pickle.load(pickle_file_Y)
    pickle_file_X.close()
    pickle_file_Y.close()

    print("train datasets X is maxtrix {0} x {1}".format(X_train.shape[0], X_train.shape[1]))
    print(X_train)
    print("train datasets Y is maxtrix {0} x {1}".format(y_train.shape[0], y_train.shape[1]))
    print(y_train)

    H, error = bagging_model_train(X_train, y_train)

    # # 绘制训练误差变化曲线
    # plt.title('训练误差的变化')
    # plt.plot(range(1, H.shape[0] + 1), error, 'o-', markersize=2)
    # plt.xlabel('基学习器个数')
    # plt.ylabel('错误率')
    # plt.show()
    #
    # # 观察结果
    # x1min, x1max = X_train[:, 0].min(), X_train[:, 0].max()
    # x2min, x2max = X_train[:, 1].min(), X_train[:, 1].max()
    # x1 = np.linspace(x1min - (x1max - x1min) * 0.2, x1max + (x1max - x1min) * 0.2, 100)
    # x2 = np.linspace(x2min - (x2max - x2min) * 0.2, x2max + (x2max - x2min) * 0.2, 100)
    # X1, X2 = np.meshgrid(x1, x2)
    #
    # for t in [3, 5, 11, 15, 20]:
    #     plt.title('前%d个基学习器' % t)
    #     plt.xlabel('xxx')
    #     plt.ylabel('xxx')

    '''预测'''
    pickle_file_test_X = open('./data/testsets_X.pkl', 'rb+')
    pickle_file_test_Y = open('./data/testsets_Y.pkl', 'rb+')
    X_test = pickle.load(pickle_file_test_X)
    y_test = pickle.load(pickle_file_test_Y)
    pickle_file_test_X.close()
    pickle_file_test_Y.close()

    print("test datasets X is maxtrix {0} x {1}".format(X_test.shape[0], X_test.shape[1]))
    print(X_test)
    print("test datasets Y is maxtrix {0} x {1}".format(y_test.shape[0], y_test.shape[1]))
    print(y_test)

    y_Est = H_predict(X_test, H)

    # 混淆矩阵
    TP = 0  # 真正例
    FN = 0  # 假反例
    FP = 0  # 假正例
    TN = 0  # 真反例

    for i in range(y_test.shape[0]):
        if y_Est[i] == 1:
            if y_Est[i] == y_test[i]:
                TP += 1
            else:
                FP += 1
        else:
            if y_Est[i] == y_test[i]:
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
