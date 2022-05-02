# coding=UTF-8
# Neural Network 神经网络模型及ABP算法（累计误差逆传播算法）实现

import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import optimize
import time

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题
np.set_printoptions(suppress=True)  # suppress=True 取消科学记数法


# 神经元阶跃函数Sigmoid
def sigmoid(x, threshold):
    # threshold = 1 x shape[1]
    y = np.zeros((x.shape[0], x.shape[1]))
    y = 1 / (1 + np.exp(-1 * (x - threshold)))
    return np.mat(y, dtype='float32')


# S型函数导数
def sigmoidGradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


# 神经元阶跃函数Sgn
def sgn(x, threshold):
    y = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] - threshold[0, j] >= 0:
                y[i, j] = 1
            else:
                y[i, j] = 0
    return np.mat(y, dtype='int')


# 均方误差计算
def mean_square(Y_calc, Y_datasets):
    # 入参Y的维度均为m x l
    E = 0.0
    temp1 = np.ones((1, Y_datasets.shape[0]))
    temp2 = np.ones((Y_datasets.shape[1], 1))
    E = (np.dot(temp1, np.dot(np.multiply(Y_calc - Y_datasets, Y_calc - Y_datasets), temp2) / 2)) / Y_datasets.shape[0]
    return E

# 正则化损失函数
def regularization_cost(Y_calc, Y_datasets, v, w, Lambda):
    m = Y_calc.shape[0]
    # 正则化向theta^2
    term = np.dot(np.transpose(np.vstack((v.reshape(-1, 1), w.reshape(-1, 1)))),
                  np.vstack((v.reshape(-1, 1), w.reshape(-1, 1))))
    ## 正则化损失函数
    J = -(np.dot(np.transpose(Y_datasets.reshape(-1, 1)), np.log(Y_calc.reshape(-1, 1))) + np.dot(
        np.transpose(1 - Y_datasets.reshape(-1, 1)), np.log(1 - Y_calc.reshape(-1, 1))) - Lambda * term / 2) / m


# 解析NN的入参，入参为一维数组，
# 构成包括：v(dxq) w(qxl) gamma(1xq) sita(1xl)
# d 输入层神经元个数
# q 隐层神经元个数
# l 输出层神经元个数，也是 Y 数据集特征个数
# v 输入->隐层 连接权值
# w 隐层->输出 连接权值
# gamma 隐层激活阈值
# sita 输出层激活阈值
# m 是 X 数据集的行数（个数），d 是 X 数据集的特征个数
def split_param(nn_Params, d, q, l):
    len = nn_Params.shape[0]
    v = nn_Params[0:d * q].reshape(d, q).copy()
    w = nn_Params[(d * q):(d * q + q * l)].reshape(q, l).copy()
    gamma = nn_Params[(d * q + q * l):(d * q + q * l + 1 * q)].reshape(1, q).copy()
    sita = nn_Params[(d * q + q * l + 1 * q):len].copy()
    return v, w, gamma, sita


def combine_param(v, w, gamma, sita):
    nn_Params = np.vstack((v.reshape(-1, 1), w.reshape(-1, 1),
                           gamma.reshape(-1, 1), sita.reshape(-1, 1)))
    return np.ravel(nn_Params)

# 随机初始化权重theta
def randInitializeWeights(L_in, L_out):
    w = np.zeros((L_out, L_in))  # 对应theta的权重
    epsilon_init = (6.0 / (L_out + L_in)) ** 0.5
    w = np.random.rand(L_out, L_in) * 2 * epsilon_init - epsilon_init
    return w


# 计算输出神经元的梯度项g，Y_row = 1 x l
# 梯度项是每个神经元的梯度项，与神经元同维度, 所以：g = 1 x l
def g_calc(pred_Y_row, Y_row):
    g = np.zeros((1, Y_row.shape[0]))
    g = np.multiply(pred_Y_row, np.multiply((1 - pred_Y_row), (Y_row - pred_Y_row)))
    return g


# 计算隐层神经元的梯度项e，输出为1 x q矩阵
# 梯度项是每个神经元的梯度项，与神经元同维度，所以：e = 1 x q
def e_calc(b_row, w, g):
    # b_row = 1 x q   w = q x l     g = 1 x l
    # g * w.T = {1 x l} * {l x q} = 1 x q
    e = np.multiply(np.multiply(b_row, (1 - b_row)), np.dot(g, w.T))
    return e


# 计算输出层神经元的连接权重变化值
def calc_delta_w(w, g, b_row, Lambda):
    # 计算每个训练例的delta_w
    # w = q x l   g = 1 x l     b = 1 x q
    delta_w = np.multiply(np.dot(b_row.reshape(-1, 1), g), Lambda)
    return delta_w


# 计算隐层神经元的连接权重变化值
def calc_delta_v(v, e, X_row, Lambda):
    # 计算每个训练例的delta_v
    # v =  d x q   e = 1 x q     X_row = 1 x d
    delta_v = np.multiply(np.dot(X_row.reshape(-1, 1), e), Lambda)
    return delta_v


# 隐层神经元激活阈值的变化值 1xq
def calc_delta_gamma(e, Lambda):
    delta_gamma = -1 * np.multiply(e, Lambda)
    return np.ravel(delta_gamma)


# 输出层神经元激活阈值的变化值 1xl
def calc_delta_sita(g, Lambda):
    delta_sita = -1 * np.multiply(g, Lambda)
    return np.ravel(delta_sita)


# 代价函数
def nnCostFunction(nn_Params, d, q, l, X, Y, Lambda):
    m = X.shape[0]
    # d 输入层神经元个数
    # q 隐层神经元个数
    # l 输出层神经元个数，也是 Y 数据集特征个数
    # v 输入->隐层 连接权值
    # w 隐层->输出 连接权值
    # gamma 隐层激活阈值
    # sita 输出层激活阈值
    # m 是 X 数据集的行数（个数），d 是 X 数据集的特征个数
    # lambda为迭代步长

    v, w, gamma, sita = split_param(nn_Params, d, q, l)

    class_Y = np.zeros((m, l))  # 数据的y对应0-9，需要映射为0/1的关系
    # 映射y
    for i in range(l):
        class_Y[:, i] = np.int32(Y == i).reshape(1, -1)  # 注意reshape(1,-1)才可以赋值

    pred_Y = np.zeros((m, l))

    # 正向传播：根据输入神经元，隐层神经元，输出神经元的连接权值和阈值，计算Y
    alpha = np.dot(X, v)  # X = m x d, v = d x q, alpha = m x q
    b = sigmoid(alpha, gamma)  # b = m x q   gamma = 1 x q
    beta = np.dot(b, w)  # beta = m x l
    pred_Y = sigmoid(beta, sita)  # Y_calc = m x l  sita = 1 x l

    print("pred_Y = {0}".format(pred_Y[0:3, :]))

    # step 3: 根据计算得出的Y值，与数据集Y值，计算累积协方差
    E = mean_square(pred_Y, class_Y)
    # E = regularization_cost(pred_Y, Y, v, w, Lambda)
    print("E = {0}".format(np.ravel(E)))
    return np.ravel(E)


# 梯度
def nnGradient(nn_Params, d, q, l, X, Y, Lambda):
    m = X.shape[0]
    # d 输入层神经元个数
    # q 隐层神经元个数
    # l 输出层神经元个数，也是 Y 数据集特征个数
    # v 输入->隐层 连接权值
    # w 隐层->输出 连接权值
    # gamma 隐层激活阈值
    # sita 输出层激活阈值
    # m 是 X 数据集的行数（个数），d 是 X 数据集的特征个数
    # lambda为迭代步长

    v, w, gamma, sita = split_param(nn_Params, d, q, l)

    class_Y = np.zeros((m, l))  # 数据的y对应0-9，需要映射为0/1的关系
    # 映射y
    for i in range(l):
        class_Y[:, i] = np.int32(Y == i).reshape(1, -1)  # 注意reshape(1,-1)才可以赋值

    pred_Y = np.zeros((m, l))

    v_grad = np.zeros((v.shape))
    w_grad = np.zeros((w.shape))
    gamma_grad = np.zeros((1, q))
    sita_grad = np.zeros((1, l))

    # 正向传播：根据输入神经元，隐层神经元，输出神经元的连接权值和阈值，计算Y
    alpha = np.dot(X, v)  # X = m x d, v = d x q, alpha = m x q
    b = sigmoid(alpha, gamma)  # b = m x q   gamma = 1 x q
    beta = np.dot(b, w)  # beta = m x l
    pred_Y = sigmoid(beta, sita)  # Y_calc = m x l  sita = 1 x l

    '''反向传播，delta为误差，'''
    delta_w = np.zeros((v.shape))
    delta_v = np.zeros((w.shape))
    for i in range(m):
        g = g_calc(pred_Y[i, :], class_Y[i, :])
        e = e_calc(b[i, :], w, g)
        delta_w = calc_delta_w(w, g, b[i, :], Lambda)
        w_grad = w_grad + delta_w
        delta_sita = calc_delta_sita(g, Lambda)
        sita_grad = sita_grad + delta_sita
        delta_v = calc_delta_v(v, e, X[i, :], Lambda)
        v_grad = v_grad + delta_v
        delta_gamma = calc_delta_gamma(e, Lambda)
        gamma_grad = gamma_grad + delta_gamma

    '''梯度'''
    grad = (np.vstack((v_grad.reshape(-1, 1), w_grad.reshape(-1, 1),
                       gamma_grad.reshape(-1, 1), sita_grad.reshape(-1, 1)))
            + np.vstack((v.reshape(-1, 1), w.reshape(-1, 1),
                         gamma.reshape(-1, 1), sita.reshape(-1, 1)))) / m
    print("grad = {0}".format(np.ravel(grad)[0:6]))
    return np.ravel(grad)

def predict(X, v, w, gamma, sita):
    ## 根据test_X, 预测Y
    m = X.shape[0]
    alpha = np.dot(X, v)  # X = m x d, v = d x q, alpha = m x q
    b = sigmoid(alpha, gamma)  # b = m x q  gamma = m x q
    beta = np.dot(b, w)  # beta = m x l
    pred_Y = sigmoid(beta, sita)  # pred_Y = m x l

    '''
    返回h中每一行最大值所在的列号
    - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
    - 最后where找到的最大概率所在的列号（列号即是对应的数字）
    '''
    # np.savetxt("h2.csv",h2,delimiter=',')
    y = np.array(np.where(pred_Y[0, :] == np.max(pred_Y, axis=1)[0]))
    for i in np.arange(1, m):
        t = np.array(np.where(pred_Y[i, :] == np.max(pred_Y, axis=1)[i]))
        y = np.vstack((y, t))
    return y


# 机器学习 P104 图5.8的具体实现
# 累计误差逆传播算法（ABP）
def NN_ABP_train(X, Y, input_cells, hidden_cells, output_cells, Lambda=10, cycles=300):
    ## Step 1: 在(0, 1)范围内随机初始化网络中所有连接权值和阈值
    v = np.random.rand(input_cells, hidden_cells)  # 隐层连接权值 V = input_cells x hidden_cells
    w = np.random.rand(hidden_cells, output_cells)  # 输出连接权值 w = hidden_cells x output_cells
    v = randInitializeWeights(input_cells, hidden_cells)  # 隐层连接权值 V = input_cells x hidden_cells
    w = randInitializeWeights(hidden_cells, output_cells)  # 输出连接权值 w = hidden_cells x output_cells
    # v = np.zeros((input_cells, hidden_cells), dtype='float32')
    # w = np.ones((hidden_cells, output_cells), dtype='float32')
    gamma = np.random.rand(1, hidden_cells)  # 隐层连接阈值 gamma = m x hidden_cells
    sita = np.random.rand(1, output_cells)  # 输出连接阈值 sita = m x output_cells
    # gamma = np.ones((1, hidden_cells), dtype='float32')
    # sita = np.ones((1, output_cells), dtype='float32')

    nn_Params = combine_param(v, w, gamma, sita)
    start = time.time()
    nn_final_Params = optimize.fmin_cg(nnCostFunction, nn_Params, fprime=nnGradient,
                                       args=(input_cells, hidden_cells, output_cells, X, Y, Lambda), maxiter=100)
    print(u'执行时间：', time.time() - start)
    print(nn_final_Params)

    pickle_file_model = open('./data/NN_ABP_Model.pkl', 'wb')
    pickle.dump(nn_final_Params, pickle_file_model)
    pickle_file_model.close()

    v, w, gamma, sita = split_param(nn_final_Params, input_cells, hidden_cells, output_cells)

    return v, w, gamma, sita


def get_NN_ABP_model(X, Y):
    input_cells = X.shape[1]  # 输入层神经元个数，= X 特征的维度
    hidden_cells = X.shape[1] + 1  # 隐层神经元个数，假定为输入层+1
    output_cells = 2  # 输出层神经元个数，= Y 特征的维度
    try:
        model_file = open('./data/NN_ABP_Model.pkl', 'rb')
        nn_final_Params = pickle.load(model_file)
        model_file.close()
        if nn_final_Params.shape[0] == 0:
            raise FileNotFoundError
        print(nn_final_Params)
        v, w, gamma, sita = split_param(nn_final_Params, input_cells, hidden_cells, output_cells)
        return v, w, gamma, sita
    except EOFError:
        print("NN ABP model file doesn't exist. Need train.")
        return NN_ABP_train(X, Y, input_cells, hidden_cells, output_cells)
    except FileNotFoundError:
        print("NN ABP model file doesn't exist. Need train.")
        return NN_ABP_train(X, Y, input_cells, hidden_cells, output_cells)


def model_evaluation(Y_predict, Y_test):
    fpr, tpr, threshold = roc_curve(Y_test, Y_predict)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(fpr, tpr)
    plt.title('ROC曲线')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    auc_score = roc_auc_score(Y_test, Y_predict)
    return auc_score


if __name__ == '__main__':
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

    v, w, gamma, sita = get_NN_ABP_model(X, Y)

    print("v:\n{0}".format(v))
    print("w:\n{0}".format(w))
    print("gamma:\n{0}".format(gamma))
    print("sita:\n{0}".format(sita))

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

    pred_Y = predict(test_X, v, w, gamma, sita)

    print(u"预测准确度为：%f%%" % np.mean(np.float64(pred_Y == test_Y.reshape(-1, 1)) * 100))


    # ## 计算预测准确率
    # print("计算预测准确率")
    # test_Y_line = test_Y.T.getA()
    # score = accuracy_score(pred_Y_array, test_Y_array)
    # print("score = {0}".format(score))
    #
    # ## 模型评估，画ROC曲线
    # print("模型评估，画ROC曲线")
    # auc_score = model_evaluation(pred_Y, test_Y)
    # print("AUC score is {0}".format(auc_score))
