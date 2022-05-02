# coding=UTF-8
# Neural Network 神经网络模型及BP算法（误差逆传播算法）实现

import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import optimize
import time

''' i 训练集角标
    m X维度角标
    j Y维度角标
    d=input_cells X维度值，输入神经元个数
    q=hidden_cells 隐层神经元个数
    l=output_cells Y纬度值，输出神经元个数
'''

np.set_printoptions(suppress=True)  # suppress=True 取消科学记数法


# 神经元阶跃函数Sigmoid
def sigmoid(Xi, threshold):
    # threshold = 1 x shape[1]
    Yi = np.zeros((1, Xi.shape[0]))
    Yi = 1 / (1 + np.exp(-1 * ((Xi + 5) - threshold)))      # 往右平移5个
    return Yi


# S型函数导数
def sigmoidGradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


# 神经元阶跃函数Sgn，入参：1 x Xi.shape[0]
def sgn(Xi, threshold):
    Yi = np.zeros((1, Xi.shape[0]))
    for j in range(Xi.shape[0]):
        if Xi[j] >= 0:
            Yi[j] = 1
        else:
            Yi[j] = 0
    return Yi


# 均方误差损失函数
def mean_square(Yi_calc, Yi_datasets):
    # 入参Y的维度均为1 x l
    E = 0.0
    temp1 = np.ones((Yi_datasets.shape[1], 1))  # l x 1
    E = np.dot(np.multiply(Yi_calc - Yi_datasets, Yi_calc - Yi_datasets), temp1) / 2
    return E


# 正则化损失函数
def regularization_cost(Yi_calc, Yi_datasets, v, w, Lambda):
    m = Yi_calc.shape[0]
    # 正则化向theta^2
    term = np.dot(np.transpose(np.vstack((v.reshape(-1, 1), w.reshape(-1, 1)))),
                  np.vstack((v.reshape(-1, 1), w.reshape(-1, 1))))
    ## 正则化损失函数
    J = -(np.dot(np.transpose(Yi_datasets.reshape(-1, 1)), np.log(Yi_calc.reshape(-1, 1))) + np.dot(
        np.transpose(1 - Yi_datasets.reshape(-1, 1)), np.log(1 - Yi_calc.reshape(-1, 1))) - Lambda * term / 2) / m


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
    nn_Params = np.ravel(nn_Params)
    len = nn_Params.shape[0]
    v = nn_Params[0:d * q].reshape(d, q).copy()
    w = nn_Params[(d * q):(d * q + q * l)].reshape(q, l).copy()
    gamma = nn_Params[(d * q + q * l):(d * q + q * l + 1 * q)].reshape(1, q).copy()
    if l == 1:
        sita = nn_Params[(d * q + q * l + 1 * q):len].copy()
    else:
        sita = nn_Params[(d * q + q * l + 1 * q):len].reshape(1, l).copy
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


# 计算输出神经元的梯度项g，Yi = 1 x l
# 梯度项是每个神经元的梯度项，与神经元同维度, 所以：g = 1 x l
def g_calc(pred_Yi, Yi):
    g = np.zeros((1, Yi.shape[0]))
    g = pred_Yi * (1 - pred_Yi) * (Yi - pred_Yi)
    return g


# 计算隐层神经元的梯度项e，输出为1 x q矩阵
# 梯度项是每个神经元的梯度项，与神经元同维度，所以：e = 1 x q
def e_calc(bi, w, g):
    # bi = 1 x q   w = q x l     g = 1 x l
    # g * w.T = {1 x l} * {l x q} = 1 x q
    e = np.multiply(np.multiply(bi, (1 - bi)), np.dot(g, w.T))
    return e


# 计算输出层神经元的连接权重变化值
def calc_delta_w(w, g, bi, Lambda):
    # 计算每个训练例的delta_w
    # w = q x l   g = 1 x l     bi = 1 x q
    delta_w = np.multiply(np.dot(bi.T, g), Lambda)
    return delta_w


# 计算隐层神经元的连接权重变化值
def calc_delta_v(v, e, Xi, Lambda):
    # 计算每个训练例的delta_v
    # v =  d x q   e = 1 x q     Xi = 1 x d
    delta_v = np.multiply(np.dot(Xi.T, e), Lambda)
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
def nnCostFunction(nn_Params, d, q, l, Xi, Yi, Lambda):
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

    # 正向传播：根据输入神经元，隐层神经元，输出神经元的连接权值和阈值，计算Y
    alphai = np.dot(Xi, v)  # X = 1 x d, v = d x q, alphai = 1 x q
    bi = sigmoid(alphai, gamma)  # bi = 1 x q   gamma = 1 x q
    betai = np.dot(bi, w)  # betai = 1 x l
    pred_Yi = sigmoid(betai, sita)  # pred_Yi = 1 x l  sita = 1 x l
    # pred_Yi = betai

    # step 3: 根据计算得出的Y值，与数据集Y值，计算累积协方差
    E = mean_square(pred_Yi, Yi)
    # E = regularization_cost(pred_Y, Y, v, w, Lambda)
    return E


# 梯度
def nnGradient(nn_Params, d, q, l, Xi, Yi, Lambda):
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

    pred_Yi = np.zeros((1, l))

    v_grad = np.zeros((v.shape))
    w_grad = np.zeros((w.shape))
    gamma_grad = np.zeros((1, q))
    sita_grad = np.zeros((1, l))

    # 正向传播：根据输入神经元，隐层神经元，输出神经元的连接权值和阈值，计算Y
    alphai = np.dot(Xi, v)  # Xi = 1 x d, v = d x q, alphai = 1 x q
    bi = sigmoid(alphai, gamma)  # b = 1 x q   gamma = 1 x q
    betai = np.dot(bi, w)  # betai = 1 x l
    pred_Yi = sigmoid(betai, sita)  # pred_Yi = 1 x l  sita = 1 x l
    # pred_Yi = betai

    '''反向传播，delta为误差，'''
    delta_w = np.zeros((v.shape))
    delta_v = np.zeros((w.shape))

    g = g_calc(pred_Yi, Yi)
    e = e_calc(bi, w, g)
    w_grad = calc_delta_w(w, g, bi, Lambda)
    sita_grad = calc_delta_sita(g, Lambda)
    v_grad = calc_delta_v(v, e, Xi, Lambda)
    gamma_grad = calc_delta_gamma(e, Lambda)

    '''梯度'''
    grad = np.ravel(np.vstack((v_grad.reshape(-1, 1), w_grad.reshape(-1, 1),
                               gamma_grad.reshape(-1, 1), sita_grad.reshape(-1, 1)))
                    + np.vstack((v.reshape(-1, 1), w.reshape(-1, 1),
                                 gamma.reshape(-1, 1), sita.reshape(-1, 1))))

    # print("grad = {0}".format(grad[0:20]))
    return grad


def predict(nn_Params, Xi, Yi):
    d = Xi.shape[1]
    q = d + 1
    l = Yi.shape[1]
    v, w, gamma, sita = split_param(nn_Params, d, q, l)
    # 正向传播：根据输入神经元，隐层神经元，输出神经元的连接权值和阈值，计算Y
    alphai = np.dot(Xi, v)  # Xi = 1 x d, v = d x q, alphai = 1 x q
    bi = sigmoid(alphai, gamma)  # b = 1 x q   gamma = 1 x q
    betai = np.dot(bi, w)  # betai = 1 x l
    pred_Yi = sigmoid(betai, sita)  # pred_Yi = 1 x l  sita = 1 x l
    # pred_Yi = betai

    return pred_Yi


# 机器学习 P104 图5.8的具体实现
# 累计误差逆传播算法（BP）
def NN_BP_train(X, Y, Lambda=1, cycles=300):
    ## Step 1: 在(0, 1)范围内随机初始化网络中所有连接权值和阈值
    input_cells = X.shape[1]  # 输入层神经元个数，= X 特征的维度
    hidden_cells = X.shape[1] + 1  # 隐层神经元个数，假定为输入层+1
    output_cells = Y.shape[1]  # 输出层神经元个数，= Y 特征的维度
    v = np.random.rand(input_cells, hidden_cells)  # 隐层连接权值 V = input_cells x hidden_cells
    w = np.random.rand(hidden_cells, output_cells)  # 输出连接权值 w = hidden_cells x output_cells
    # v = randInitializeWeights(input_cells, hidden_cells)      # 隐层连接权值 V = input_cells x hidden_cells
    # w = randInitializeWeights(hidden_cells, output_cells)     # 输出连接权值 w = hidden_cells x output_cells
    gamma = np.random.rand(1, hidden_cells)  # 隐层连接阈值 gamma = m x hidden_cells
    sita = np.random.rand(1, output_cells)  # 输出连接阈值 sita = m x output_cells
    # gamma = np.ones((1, hidden_cells), dtype='float32')
    # sita = np.ones((1, output_cells), dtype='float32')

    start = time.time()
    nn_Params = combine_param(v, w, gamma, sita)
    i = 0
    pred_Y = np.zeros((Y.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        # i = 7
        Xi = X[i, :].reshape(1, X.shape[1])
        Yi = Y[i, :].reshape(1, Y.shape[1])
        print("X[{0}] = {1}".format(i, Xi))
        print("Y[{0}] = {1}".format(i, Yi))
        nn_Params = optimize.fmin_cg(nnCostFunction, nn_Params, fprime=nnGradient,
                                     args=(input_cells, hidden_cells, output_cells, Xi, Yi, Lambda), maxiter=1000)

        print("trained nn_Params = {1}".format(i, nn_Params[0:20]))
        pred_Yi = predict(nn_Params, Xi, Yi)
        print("predicted Y[{0}] = {1}".format(i, pred_Yi))
        pred_Y[i, :] = np.ravel(pred_Yi)

    pickle_file_model = open('./data/NN_BP_Model.pkl', 'wb')
    pickle.dump(nn_Params, pickle_file_model)
    pickle_file_model.close()

    v, w, gamma, sita = split_param(nn_Params, input_cells, hidden_cells, output_cells)

    print(u'执行时间：', time.time() - start)

    return v, w, gamma, sita


def get_NN_BP_model(X, Y):
    input_cells = X.shape[1]  # 输入层神经元个数，= X 特征的维度
    hidden_cells = X.shape[1] + 1  # 隐层神经元个数，假定为输入层+1
    output_cells = Y.shape[1]  # 输出层神经元个数，= Y 特征的维度
    try:
        model_file = open('./data/NN_BP_Model.pkl', 'rb')
        nn_final_Params = pickle.load(model_file)
        model_file.close()
        if nn_final_Params.shape[0] == 0:
            raise FileNotFoundError
        print(nn_final_Params)
        v, w, gamma, sita = split_param(nn_final_Params, input_cells, hidden_cells, output_cells)
        return v, w, gamma, sita
    except EOFError:
        print("NN ABP model file doesn't exist. Need train.")
        return NN_BP_train(X, Y)
    except FileNotFoundError:
        print("NN ABP model file doesn't exist. Need train.")
        return NN_BP_train(X, Y)


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

    v, w, gamma, sita = get_NN_BP_model(X, Y)
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

    pred_Y = np.zeros((test_Y.shape[0], test_Y.shape[1]), dtype='int')

    ## 根据test_X, 预测Y
    for i in range(test_X.shape[0]):
        alpha = np.dot(test_X[i, :], v)  # X = 1 x d, v = d x q, alpha = 1 x q
        b = sigmoid(alpha - gamma)  # gamma = 1 x q
        beta = np.dot(b, w)  # beta = 1 x l
        calc_Y = sgn(beta - sita)  # sita = 1 x l  pred_Y = 1 x l

        pred_Y[i, :] = calc_Y

    print(pred_Y[0:100])
    print(test_Y[0:100])

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
