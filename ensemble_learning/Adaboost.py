# coding=UTF-8
# 集成学习 AdaBoost算法的实现，西瓜书图8.2

import numpy as np
import matplotlib.pyplot as plt
import pickle

np.set_printoptions(suppress=True)  # suppress=True 取消科学记数法


def loadSimpData():
    """
    创建单层决策树的数据集
    parameters:
        无
    return:
        dataMat - 有两个特征的数据矩阵
        classLabels - 数据标签
    """
    dataMat = np.matrix([[1., 2.1],
                         [1.5, 1.6],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]  # 正例1，反例-1
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, inequal):
    """
    单层决策树(即只能用一个属性进行一次分类)分类函数
    :param dataMatrix: 数据特征矩阵
    :param dimen:  第dimen列，也就是第几个特征
    :param threshVal: 阈值
    :param inequal: 标志
    :return:
        retArry: 分类结果
    """
    retArray = np.ones((dataMatrix.shape[0], 1))  # 初始化retArry为1(假设全为正例)
    # 注：此处的等号表示样本点恰在阈值线上，此处假设在此弱分类器中阈值线上的样本是反例
    # (当然也可以设为正例，但是要注意训练样本和测试样本的分类函数的这个等号要一致)
    if inequal == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 如果小于阈值，则赋值为-1(反例)
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0  # 如果大于阈值，则赋值为-1
    return retArray


def buildStump(dataset, classLabels, D):
    """
    找到数据集上最佳的单层决策树--单层决策树是指只考虑其中一个特征，用该特征进行分类，x=阈值(向量只有一个x轴没有y轴，那么用一条垂直于x轴的线进行分类即可)
    例如本文例子，如果已第一列特征为基础，阈值选择X=1.3这条竖线，并设置>1.3的为反例，<1.3的为正例，这样就构造了一个二分类器
    :param dataset: 数据特征矩阵
    :param classLabels: 数据标签
    :param D: 样本权重
    :return:
        bestStump: 保存单个最优弱分类器的信息的(第几个特征，分类的阈值，lt还是gt，此弱分类器的权重alpha)
        minOverallError: 最小误差(弱分类器权重计算中的误差，即西瓜书p174图8.3中第4行的那个误差)
        bestClassEst: 保存最佳的分类结果，即西瓜书p173式(8.4)中的ht(x)
    """
    dataMatrix = np.mat(dataset)
    labelMat = np.mat(classLabels)  # 将列表转换为向量
    m, n = dataMatrix.shape
    numSteps = 10.0  # 总步数，计算步长用的
    bestStump = {}  # 用来保存单个最优弱分类器的信息的(第几个特征，分类的阈值，lt还是gt，此弱分类器的权重alpha)
    bestClasEst = np.mat(np.zeros((m, 1)))  # 保存最佳的分类结果
    minOverallError = float('inf')  # 最小总误差初始化为正无穷大
    for i in range(n):  # 分别对每个特征计算最优的划分阈值(分别对每个特征求其最小的总误差，得到最小总误差最小的那个特征，此特征被选为分类特征)
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长---阈值递增的步长
        for j in range(-1, int(numSteps) + 1):
            threshVal = (rangeMin + float(j) * stepSize)  # 根据步长计算阈值
            # lt:less than 是指在该阈值下，如果<阈值，则分类为-1
            # gt:greater than是指在该阈值下,如果>阈值，则分类为1，
            # 有两种情况，一种是阈值线的右边是正例(gt)，左边是反例(lt)；另一种是阈值线的左边是正例，右边是反例(gt)，所以每个阈值要计算两种情况的误差
            for inequal in ['lt', 'gt']:
                # 计算分类结果，即若以当前threshVal为阈值分类，那么此时的训练样本分类结果如何（1表示正例，-1表示反例）
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵(不是保存误差的，而是用来保存哪些样本分类错误，哪些样本分类正确)
                # 若分类正确的,则记为0，否则记为1，下面乘该样本的权重当作误差，列表之间可以直接判断相等
                errArr[predictedVals == labelMat] = 0
                # 基于权重向量D而不是其他错误计算指标来评价分类器的，不同的分类器计算方法不同
                overallError = D.T * errArr
                # print("第%d个特征, 阈值为%.2f, ineqal: %s, 该阈值的决策树对所有样本的总误差为%.3f" % (i, threshVal, inequal, overallError))
                if overallError < minOverallError:  # 找到总误差最小的分类方式--找到当前最好的弱分类器
                    minOverallError = overallError
                    bestClasEst = predictedVals.copy()  # 保存该阈值的分类结果
                    bestStump['dim'] = i  # 保存特征
                    bestStump['thresh'] = threshVal  # 保存最优阈值
                    bestStump['ineq'] = inequal  # 保存是lt还是gt

    return bestStump, minOverallError, bestClasEst


def adaBoostTrainDS(dataset, classLabels, numIt=100):
    """
    adaboost算法核心代码
    :param dataset: 数据特征矩阵
    :param classLabels: 数据标签
    :param numIt: 最大迭代次数(每迭代一次生成一个弱分类器，虽然设置的是40个，但若迭代过程中不满40次误差就为0时就可以停止迭代了，说明用不了40个弱分类器就可以完全正确分类)
    :return:
        weakClassifiterArr: 训练好的分类器
        aggClassEst: 加权投票（p173式(8.4) 对加权投票值取sign函数就可以得到预测值）
    """
    weakClassifiterArr = []  # 保存多个训练好的弱学习器
    m = dataset.shape[0]  # 行数(样本个数)
    D = np.mat(np.ones((m, 1)) / m)  # 初始化每个样本的权重(均是1/m),即p173式(8.5)中的D(xi)
    aggClassEst = np.mat(np.zeros((m, 1)))  # 保存每一轮累加的投票值(初始化为0)，后面最终判断某一区域是正例还是反例要用(对加权投票套sign()函数)
    for i in range(numIt):
        # p174图8.3第3行 ~ 第4行
        # bestStump格式为：{'dim':最优分类列; 'thresh':该列最优分类阈值; 'ineq':阈值右正还是左正; 'alpha': 本轮弱学习器权重}
        bestStump, error, bestClasEst = buildStump(dataset, classLabels, D)  # 构建单个单层决策树
        # if error > 0.5: break    # 检查当前基分类器是否弱于随机猜测，如果弱于，则放弃该基学习器
        # p174图8.3第6行  计算弱学习算法权重alpha，使error不等于0，因为分母不能为0(注意此权重是弱学习器的权重而非是单个样本的权重)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha  # 存储本轮弱学习器权重
        print("第%d次迭代中得到的最优单层决策树：第%d个特征, 阈值为%.2f, ineqal: %s, 该阈值的决策树对所有样本的总误差为%.3f, 此弱分类器的权重为：%.3f" %
              (i, bestStump['dim'], bestStump['thresh'], bestStump['ineq'], error, bestStump['alpha']))
        weakClassifiterArr.append(bestStump)  # 存储单层决策树
        # print("bestClasEst:"bestClasEst.T)
        # 下面三行代码是：西瓜书p174图8.3第7行  根据样本权重公式，更新样本权重
        expon = np.multiply(-1 * alpha * np.mat(classLabels), bestClasEst)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # 计算AdaBoost误差，当误差为0的时候，退出循环
        # p173式(8.4)  加权投票（对加权投票值取sign函数就可以得到预测值），注意这里包括了目前已经训练好的每一个弱分类器，线性组合
        aggClassEst += alpha * bestClasEst
        print("前{0}个弱分类器得到的aggClassEst:{1} ".format(i, aggClassEst.T))
        temp = np.sign(aggClassEst) != np.mat(classLabels)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels),
                                np.ones((m, 1)))  # 计算误差     aggErrors向量中元素为1的表示分错的样本，为0的表示分对的样本
        # np.sign(aggClassEst) != np.mat(classLabels).T也可以写ClassEst != np.mat(classLabels).T，表示分类错了则为true，分类对了则为false,自动转换成0和1，
        errorRate = aggErrors.sum() / m  # aggErrors.sum()就表示为总共有多少个样本分类错误
        print("分错样本个数/样本总个数: ", errorRate)
        if errorRate == 0.0: break  # 误差为0，说明样本被完全正确的分类了，不再需要更多的弱学习器了，退出循环
    return weakClassifiterArr


def showDataSet(dataMat, labelMat, weakClassifiterArr):
    """
    数据可视化
    :param dataMat: 数据特征矩阵
    :param labelMat: 数据标签
    :param weakClassifiterArr: 训练好的弱分类器集合
    :return: 无
    """
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):  # 正负样本分类
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)
    # 绘制样本
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1],
                c='red')  # 正样本   np.transpose(data_plus_np)[0]表示data_plus_np转置后的第0行
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], c='green')  # 负样本

    # 绘制训练函数图像
    for i in range(len(weakClassifiterArr)):  # 每个弱分类器一条线(一个阈值)
        if weakClassifiterArr[i]['thresh'] > 20:
            continue
        if weakClassifiterArr[i]['dim'] == 0:  # 如果分类特征是第0个特征(x1)
            x2 = np.arange(1.0, 3.0,
                           1)  # x1是一个2维列表[1,2]    arange()函数用法：https://blog.csdn.net/island1995/article/details/90179076
            plt.plot([weakClassifiterArr[i]['thresh'], weakClassifiterArr[i]['thresh']], x2)  # 因为确定一条线至少要两个点，所以至少都是二维列表
        else:  # 如果分类特征是第1个特征(x2)
            x1 = np.arange(1.0, 3.0, 1)
            plt.plot(x1, [weakClassifiterArr[i]['thresh'], weakClassifiterArr[i]['thresh']])

    plt.title('Training sample data')  # 绘制title
    # 绘制坐标轴
    plt.xlabel('x1')  # 第0个特征
    plt.ylabel('x2')  # 第1个特征
    plt.show()


def adaClassify(testSample, weakClassifiterArr):
    """
    AdaBoost分类函数
    Parameters:
        testSample - 待分类样例(测试样本集合)
        weakClassifiterArr - 训练好的分类器集合
    Returns:
        分类结果
    """
    dataMatrix = np.mat(testSample)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(weakClassifiterArr)):  # 遍历所有分类器，进行分类
        bestClasEst = stumpClassify(dataMatrix, weakClassifiterArr[i]['dim'], weakClassifiterArr[i]['thresh'],
                                    weakClassifiterArr[i]['ineq'])
        aggClassEst += weakClassifiterArr[i]['alpha'] * bestClasEst  # 加权投票 p173式(8.4)
    print("测试样本的加权投票为：", aggClassEst)
    return np.sign(aggClassEst)  # p174图8.3的最后一行“输出”


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

    weakClassifiterArr = adaBoostTrainDS(X_train, y_train)  # 通过adaboost得到多个弱分类器，保存在weakClassifiterArr列表中
    showDataSet(X_train, y_train, weakClassifiterArr)  # 画图

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

    y_Est = adaClassify(X_test, weakClassifiterArr)

    # 混淆矩阵
    TP = 0  # 真正例
    FN = 0  # 假反例
    FP = 0  # 假正例
    TN = 0  # 真反例

    # 通过后验概率，预测Y ~ （0,1）
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
