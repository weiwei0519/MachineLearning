# coding=UTF-8
# Neural Network 神经网络模型及ABP算法（累计误差逆传播算法）实现

import numpy as np

# 神经元阶跃函数Sigmoid
def sigmoid(x, threshold):
    # threshold = 1 x shape[1]
    y = np.zeros((x.shape[0], x.shape[1]))
    y = 1 / (1 + np.exp(-1 * (x - threshold)))
    return np.mat(y, dtype='float32')

def split_param(nn_Params, d, q, l):
    len = nn_Params.shape[0]
    v = nn_Params[0:d * q].reshape(d, q).copy()
    w = nn_Params[(d * q):(d * q + q * l)].reshape(q, l).copy()
    gamma = nn_Params[(d * q + q * l):(d * q + q * l + 1 * q)].reshape(1, q).copy()
    sita = nn_Params[(d * q + q * l + 1 * q):len].reshape(1, l).copy()
    return v, w, gamma, sita

def combine_param(v, w, gamma, sita):
    nn_Params = np.vstack((v.reshape(-1, 1), w.reshape(-1, 1),
                       gamma.reshape(-1, 1), sita.reshape(-1, 1)))
    return np.ravel(nn_Params)


if __name__ == '__main__':

    x = '1'
    if x > 0:
        print('true')

    # v = 1 + np.arange(14 * 15).reshape(14, 15)/1000
    # w = 2 + np.arange(15 * 2).reshape(15, 2)/100
    # gamma = 3 + np.arange(1 * 15).reshape(1, 15)/100
    # sita = 4 + np.arange(1 * 2).reshape(1, 2)/10
    #
    # # print(v)
    # # print(w)
    # # print(gamma)
    # # print(sita)
    #
    # # print(combine_param(v, w, gamma, sita))
    #
    # nn_Params = combine_param(v, w, gamma, sita)
    #
    # nn_Params = []
    #
    # v, w, gamma, sita = split_param(nn_Params, 14, 15, 2)
    #
    # print(v)
    # print(w)
    # print(gamma)
    # print(sita)


