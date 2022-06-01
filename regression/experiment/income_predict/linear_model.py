# coding=UTF-8
# 线性回归模型参数训练

import pickle
import numpy as np
from scipy.optimize import minimize

pickle_file_X = open('./data/trainsets_X.pkl', 'rb+')
pickle_file_Y = open('./data/trainsets_Y.pkl', 'rb+')
X = pickle.load(pickle_file_X)
Y = pickle.load(pickle_file_Y)
pickle_file_X.close()
pickle_file_Y.close()

print("datasets X is maxtrix {0} x {1}".format(X.shape[0], X.shape[1]))
print(X)
print("datasets Y is maxtrix {0} x {1}".format(Y.shape[0], Y.shape[1]))
print(Y)

N = np.size(Y)

Y1 = np.array([Y[i] for i in range(Y.shape[0]) if Y[i] == 1])
Y0 = np.array([Y[i] for i in range(Y.shape[0]) if Y[i] == 0])
train_radio = len(Y1) / len(Y0)


def fun(beta):
    print(beta)
    sum = 0
    for j in range(N):
        # 机器学习（西瓜书） Page59 公式3.27
        sum += (-Y[j] * np.dot(beta, X[j].T) + np.log(1 + np.exp(np.dot(beta, X[j].T))))
    print(sum)
    return sum


def fun_jac(beta):
    jac = np.zeros(np.shape(beta), dtype=np.double)
    p1 = np.zeros(N, dtype=np.double)
    for j in range(N):
        p1[j] = np.exp(np.dot(beta, X[j].T)) / (1 + np.exp(np.dot(beta, X[j].T)))
        jac = jac - (X[j]) * (Y[j] - p1[j])
    return jac


def fun_hess(beta):
    hess = np.zeros((np.size(beta), np.size(beta)), dtype=np.double)
    p1 = np.zeros(N, dtype=np.double)
    for j in range(N):
        p1[j] = np.exp(np.dot(beta, X[j].T)) / (1 + np.exp(np.dot(beta, X[j].T)))
        hess += np.dot(X[j], X[j].T) * p1[j] * (1 - p1[j])
    return hess


def callback(xk):
    print(xk)


def line(beta, x):
    return 1 / beta[1] * (- beta[0] * x - beta[2])


# y[i] = f(x[i]) = wx[i] + b
# 机器学习 Page58~59 公式（3.19, 3.7, 3.8）
# logistic_regression 对数几率回归模型。
# 用极大似然法估计参数 W 和 B，公式见：3.27
def get_logit_regression_model(X, Y):
    model = {}
    try:
        model_file = open('./data/logit_regression_model.pkl', 'rb')
        model = pickle.load(model_file)
        model_file.close()
        if len(model.items()) == 0:
            raise FileNotFoundError
        print(model)
        return model
    except EOFError:
        print("logit_regression_model file doesn't exist. Need train.")
        return logit_regression_train(X, Y)
    except FileNotFoundError:
        print("logit_regression_model file doesn't exist. Need train.")
        return logit_regression_train(X, Y)

    # [[10639.08081222]]
    # Optimization terminated successfully.    (Exit mode 0)
    #             Current function value: 10639.08081222017
    #             Iterations: 24
    #             Function evaluations: 446
    #             Gradient evaluations: 24
    #      fun: 10639.08081222017
    #      jac: array([ 0.00634766,  0.00146484,  0.00256348,  0.00292969,  0.01403809,
    #         0.00073242,  0.00097656,  0.0032959 ,  0.00036621,  0.01208496,
    #         0.00317383,  0.00268555,  0.00793457, -0.00219727,  0.00671387])
    #  message: 'Optimization terminated successfully.'
    #     nfev: 446
    #      nit: 24
    #     njev: 24
    #   status: 0
    #  success: True
    #        x: array([-2.94973433e+00, -7.21878460e-02, -7.89246596e-01,  2.90523852e-02,
    #        -5.66677407e+00,  5.92249319e+00,  4.35204444e-01,  1.72783897e+00,
    #         2.88604275e-01, -1.00499803e+00, -3.22034239e+01, -2.36590794e+00,
    #        -2.89880104e+00,  6.36351368e-01,  6.15354613e+00])

    return model


def logit_regression_train(X, Y):
    beta0 = np.ones((1, X.shape[1]), dtype=np.double)
    model = minimize(fun, beta0, method='SLSQP', callback=callback, tol=1.e-6,
                     options={'disp': True, 'maxiter': 50})
    # save model
    model_file = open('./data/logit_regression_model.pkl', 'wb')
    pickle.dump(model, model_file)
    model_file.close()
    return model


if __name__ == '__main__':
    logit_regre_model = get_logit_regression_model(X, Y)
    beta_optimized = logit_regre_model.x

    # model test, load testsets.
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

    test_N = np.size(test_Y)

    predict_Y = np.zeros((test_N, 1), dtype=np.int8)

    # 混淆矩阵
    TP = 0  # 真正例
    FN = 0  # 假反例
    FP = 0  # 假正例
    TN = 0  # 真反例

    # 通过后验概率，预测Y ~ （0,1）
    for i in range(test_N):
        P_1 = np.exp(np.dot(beta_optimized, X[i].T)) / (1 + np.exp(np.dot(beta_optimized, X[i].T)))
        P_0 = 1 - P_1
        predict_radio = P_1 / P_0
        if train_radio < 5:
            # 训练样本中，只有当正例与反例样本数极不平衡时，才用 Y/(1-Y) > Y1/Y0来预测是否正例，否则用Y/(1-Y)>1判断
            train_radio = 1
        if predict_radio > train_radio:
            predict_Y[i] = 1
            if test_Y[i] == predict_Y[i]:
                TP += 1
            else:
                FP += 1
        else:
            predict_Y[i] = 0
            if test_Y[i] == predict_Y[i]:
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

    # 真正例TP = 9268 / 假正例FP = 3019 / 假反例FN = 2092 / 真反例TN = 681
    # 查准率P = 0.7542931553674616 / 查全率R = 0.8158450704225352

# scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
#
# 其中：
# fun：目标函数，返回单值，
# x0：初始迭代值，
# args：要输入到目标函数中的参数
# method：求解的算法，目前可选的有
#         ‘Nelder-Mead’
#         ‘Powell’
#         ‘CG’
#         ‘BFGS’
#         ‘Newton-CG’
#         ‘L-BFGS-B’
#         ‘TNC’
#         ‘COBYLA’
#         ‘SLSQP’           梯度下降算法
#         ‘dogleg’
#         ‘trust-ncg’
#         以及在 version 0.14.0，还能自定义算法
#         以上算法的解释和相关用法见 minimize 函数的官方说明文档，一般求极值多用 'SLSQP'算法
# jac：目标函数的雅可比矩阵。可选项，仅适用于CG，BFGS，Newton-CG，L-BFGS-B，TNC，SLSQP，dogleg，trust-ncg。如果jac是布尔值并且为True，则假定fun与目标函数一起返回梯度。如果为False，将以数字方式估计梯度。jac也可以返回目标的梯度。此时，它的参数必须与fun相同。
# hess，hessp：可选项，目标函数的Hessian（二阶导数矩阵）或目标函数的Hessian乘以任意向量p。仅适用于Newton-CG，dogleg，trust-ncg。
# bounds：可选项，变量的边界（仅适用于L-BFGS-B，TNC和SLSQP）。以（min，max）对的形式定义 x 中每个元素的边界。如果某个参数在 min 或者 max 的一个方向上没有边界，则用 None 标识。如（None, max）
# constraints：约束条件（只对 COBYLA 和 SLSQP）。dict 类型。
#     type : str， ‘eq’ 表示等于0，‘ineq’ 表示不小于0
#     fun : 定义约束的目标函数
#     jac : 函数的雅可比矩阵 (只用于 SLSQP)，可选项。
#     args : fun 和 雅可比矩阵的入参，可选项。
# tol：迭代停止的精度。 ‘tol’是指迭代之间目标值dJ的最小可接受变化
# callback(xk)：每次迭代要回调的函数，需要有参数 xk
# options：其他选项
#     maxiter :  最大迭代次数
#     disp :  是否显示过程信息
# 以上参数更具体的介绍见官网相关页面。
