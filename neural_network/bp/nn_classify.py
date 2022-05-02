# coding=UTF-8
# Neural Network 神经网络模型及BP算法（误差逆传播算法）实现

import numpy as np
import pickle
from scipy import io as spio
from matplotlib import pyplot as plt
from scipy import optimize
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题
np.set_printoptions(suppress=True)  # suppress=True 取消科学记数法

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import time


def neuralNetwork(input_layer_size, hidden_layer_size, out_put_layer, X, y):
    '''scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)'''

    m, n = X.shape
    """digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    m,n = X.shape

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)"""

    ##　随机显示几行数据
    rand_indices = [t for t in [np.random.randint(x - x, m) for x in range(100)]]  # 生成100个0-m的随机数

    # nn_params = np.vstack((Theta1.reshape(-1,1),Theta2.reshape(-1,1)))

    Lambda = 10

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, out_put_layer)

    initial_nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))  # 展开theta
    # np.savetxt("testTheta.csv",initial_nn_params,delimiter=",")
    start = time.time()
    result = optimize.fmin_cg(nnCostFunction, initial_nn_params, fprime=nnGradient,
                              args=(input_layer_size, hidden_layer_size, out_put_layer, X, y, Lambda), maxiter=300)
    print(u'执行时间：', time.time() - start)
    print(result)
    '''可视化 Theta1'''
    length = result.shape[0]
    Theta1 = result[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = result[hidden_layer_size * (input_layer_size + 1):length].reshape(out_put_layer, hidden_layer_size + 1)

    pickle_file_model = open('./data/NN_classify_Model.pkl', 'wb')
    pickle.dump(result, pickle_file_model)
    pickle_file_model.close()

    return Theta1, Theta2


# 代价函数
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    length = nn_params.shape[0]  # theta的中长度
    # 还原theta1和theta2
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):length].reshape(num_labels, hidden_layer_size + 1)

    # np.savetxt("Theta1.csv",Theta1,delimiter=',')

    m = X.shape[0]
    class_y = np.zeros((m, num_labels))  # 数据的y对应0-9，需要映射为0/1的关系
    # 映射y
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(1, -1)  # 注意reshape(1,-1)才可以赋值

    '''去掉theta1和theta2的第一列，因为正则化时从1开始'''
    Theta1_colCount = Theta1.shape[1]
    Theta1_x = Theta1[:, 1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]
    Theta2_x = Theta2[:, 1:Theta2_colCount]
    # 正则化向theta^2
    term = np.dot(np.transpose(np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1)))),
                  np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1))))

    '''正向传播,每次需要补上一列1的偏置bias'''
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, np.transpose(Theta2))
    h = sigmoid(z3)
    '''代价'''  ## 正则化损失函数
    J = -(np.dot(np.transpose(class_y.reshape(-1, 1)), np.log(h.reshape(-1, 1))) + np.dot(
        np.transpose(1 - class_y.reshape(-1, 1)), np.log(1 - h.reshape(-1, 1))) - Lambda * term / 2) / m
    J = np.ravel(J)
    print("Cost = {0}".format(J))
    # temp1 = (h.reshape(-1,1)-class_y.reshape(-1,1))
    # temp2 = (temp1**2).sum()
    # J = 1/(2*m)*temp2
    return J


# 梯度
def nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size,
                                                                             input_layer_size + 1).copy()  # 这里使用copy函数，否则下面修改Theta的值，nn_params也会一起修改
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):length].reshape(num_labels,
                                                                                  hidden_layer_size + 1).copy()
    m = X.shape[0]
    class_y = np.zeros((m, num_labels))  # 数据的y对应0-9，需要映射为0/1的关系
    # 映射y
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(1, -1)  # 注意reshape(1,-1)才可以赋值

    '''去掉theta1和theta2的第一列，因为正则化时从1开始'''
    Theta1_colCount = Theta1.shape[1]
    Theta1_x = Theta1[:, 1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]
    Theta2_x = Theta2[:, 1:Theta2_colCount]

    Theta1_grad = np.zeros((Theta1.shape))  # 第一层到第二层的权重
    Theta2_grad = np.zeros((Theta2.shape))  # 第二层到第三层的权重

    '''正向传播，每次需要补上一列1的偏置bias'''
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, np.transpose(Theta2))
    h = sigmoid(z3)

    '''反向传播，delta为误差，'''
    delta3 = np.zeros((m, num_labels))
    delta2 = np.zeros((m, hidden_layer_size))
    for i in range(m):
        # delta3[i,:] = (h[i,:]-class_y[i,:])*sigmoidGradient(z3[i,:])  # 均方误差的误差率
        delta3[i, :] = h[i, :] - class_y[i, :]  # 交叉熵误差率
        Theta2_grad = Theta2_grad + np.dot(np.transpose(delta3[i, :].reshape(1, -1)), a2[i, :].reshape(1, -1))
        delta2[i, :] = np.dot(delta3[i, :].reshape(1, -1), Theta2_x) * sigmoidGradient(z2[i, :])
        Theta1_grad = Theta1_grad + np.dot(np.transpose(delta2[i, :].reshape(1, -1)), a1[i, :].reshape(1, -1))

    Theta1[:, 0] = 0
    Theta2[:, 0] = 0
    '''梯度'''
    grad = (np.vstack((Theta1_grad.reshape(-1, 1), Theta2_grad.reshape(-1, 1))) + Lambda * np.vstack(
        (Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))) / m
    return np.ravel(grad)


# S型函数
def sigmoid(z):
    h = np.zeros((len(z), 1))  # 初始化，与z的长度一致

    h = 1.0 / (1.0 + np.exp(-z))
    return h


# S型函数导数
def sigmoidGradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


# 随机初始化权重theta
def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))  # 对应theta的权重
    epsilon_init = (6.0 / (L_out + L_in)) ** 0.5
    W = np.random.rand(L_out,
                       1 + L_in) * 2 * epsilon_init - epsilon_init  # np.random.rand(L_out,1+L_in)产生L_out*(1+L_in)大小的随机矩阵
    return W


# 检验梯度是否计算正确
def checkGradient(Lambda=0):
    '''构造一个小型的神经网络验证，因为数值法计算梯度很浪费时间，而且验证正确后之后就不再需要验证了'''
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    initial_Theta1 = debugInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = debugInitializeWeights(hidden_layer_size, num_labels)
    X = debugInitializeWeights(input_layer_size - 1, m)
    y = np.transpose(np.mod(np.arange(1, m + 1), num_labels))  # 初始化y

    y = y.reshape(-1, 1)
    nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))  # 展开theta
    '''BP求出梯度'''
    grad = nnGradient(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, Lambda)
    '''使用数值法计算梯度'''
    num_grad = np.zeros((nn_params.shape[0]))
    step = np.zeros((nn_params.shape[0]))
    e = 1e-4
    for i in range(nn_params.shape[0]):
        step[i] = e
        loss1 = nnCostFunction(nn_params - step.reshape(-1, 1), input_layer_size, hidden_layer_size,
                               num_labels, X, y,
                               Lambda)
        loss2 = nnCostFunction(nn_params + step.reshape(-1, 1), input_layer_size, hidden_layer_size,
                               num_labels, X, y,
                               Lambda)
        num_grad[i] = (loss2 - loss1) / (2 * e)
        step[i] = 0
    # 显示两列比较
    res = np.hstack((num_grad.reshape(-1, 1), grad.reshape(-1, 1)))
    print("检查梯度的结果，第一列为数值法计算得到的，第二列为BP得到的:")
    print(res)


# 初始化调试的theta权重
def debugInitializeWeights(fan_in, fan_out):
    W = np.zeros((fan_out, fan_in + 1))
    x = np.arange(1, fan_out * (fan_in + 1) + 1)
    W = np.sin(x).reshape(W.shape) / 10
    return W


def get_NN_ABP_model(input_layer_size, hidden_layer_size, num_labels, X, y):
    # input_layer_size 输入层神经元个数，= X 特征的维度
    # hidden_layer_size 隐层神经元个数，假定为输入层+1
    # num_labels 输出层神经元个数，= Y 特征的维度
    try:
        model_file = open('./data/NN_classify_Model.pkl', 'rb')
        nn_params = pickle.load(model_file)
        model_file.close()
        if nn_params.shape[0] == 0:
            raise FileNotFoundError
        print(nn_params)
        length = nn_params.shape[0]
        Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size,
                                                                                 input_layer_size + 1).copy()  # 这里使用copy函数，否则下面修改Theta的值，nn_params也会一起修改
        Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):length].reshape(num_labels,
                                                                                      hidden_layer_size + 1).copy()
        return Theta1, Theta2
    except EOFError:
        print("NN_classify model file doesn't exist. Need train.")
        return neuralNetwork(input_layer_size, hidden_layer_size, num_labels, X, y)
    except FileNotFoundError:
        print("NN_classify model file doesn't exist. Need train.")
        return neuralNetwork(input_layer_size, hidden_layer_size, num_labels, X, y)


# 预测
def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    # p = np.zeros((m,1))
    '''正向传播，预测结果'''
    X = np.hstack((np.ones((m, 1)), X))
    h1 = sigmoid(np.dot(X, np.transpose(Theta1)))
    h1 = np.hstack((np.ones((m, 1)), h1))
    h2 = sigmoid(np.dot(h1, np.transpose(Theta2)))

    '''
    返回h中每一行最大值所在的列号
    - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
    - 最后where找到的最大概率所在的列号（列号即是对应的数字）
    '''
    # np.savetxt("h2.csv",h2,delimiter=',')
    p = np.array(np.where(h2[0, :] == np.max(h2, axis=1)[0]))
    for i in np.arange(1, m):
        t = np.array(np.where(h2[i, :] == np.max(h2, axis=1)[i]))
        p = np.vstack((p, t))
    return p


if __name__ == "__main__":
    # checkGradient()
    pickle_file_X = open('./data/trainsets_X.pkl', 'rb+')
    pickle_file_Y = open('./data/trainsets_Y.pkl', 'rb+')
    X = pickle.load(pickle_file_X)
    y = pickle.load(pickle_file_Y)
    pickle_file_X.close()
    pickle_file_Y.close()

    print("train datasets X is maxtrix {0} x {1}".format(X.shape[0], X.shape[1]))
    print(X)
    print("train datasets Y is maxtrix {0} x {1}".format(y.shape[0], y.shape[1]))
    print(y)

    Theta1, Theta2 = get_NN_ABP_model(14, 5, 2, X, y)

    '''预测'''
    pickle_file_test_X = open('./data/testsets_X.pkl', 'rb+')
    pickle_file_test_Y = open('./data/testsets_Y.pkl', 'rb+')
    X = pickle.load(pickle_file_test_X)
    y = pickle.load(pickle_file_test_Y)
    pickle_file_test_X.close()
    pickle_file_test_Y.close()

    print("test datasets X is maxtrix {0} x {1}".format(X.shape[0], X.shape[1]))
    print(X)
    print("test datasets Y is maxtrix {0} x {1}".format(y.shape[0], y.shape[1]))
    print(y)
    p = predict(Theta1, Theta2, X)
    print(u"预测准确度为：%f%%" % np.mean(np.float64(p == y.reshape(-1, 1)) * 100))
    res = np.hstack((p, y.reshape(-1, 1)))
    np.savetxt("predict.csv", res, delimiter=',')
