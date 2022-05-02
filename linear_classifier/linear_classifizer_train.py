# coding=UTF-8
# 回归（Regression）与最小二乘法（Least squares）
# 模型训练

from scipy.optimize import minimize
import numpy as np
import pickle
import matplotlib.pyplot as plt

pickle_file = open('data_x.pkl', 'rb')
data_x = pickle.load(pickle_file)
pickle_file.close()

pickle_file2 = open('data_y.pkl', 'rb')
data_y = pickle.load(pickle_file2)
pickle_file2.close()

print(data_x)
print(data_y)

N = np.size(data_y)


def fun(beta):
    sum = 0
    for j in range(N):
        # 机器学习（西瓜书） Page59 公式3.27
        print("data_y[{0}]".format(j))
        print(data_y[j])
        print("data_x[{0}]".format(j))
        print(data_x[j])
        sum += (-data_y[j] * np.dot(beta, data_x[j].T) + np.log(1 + np.exp(np.dot(beta, data_x[j].T))))
    return sum


def fun_jac(beta):
    jac = np.zeros(np.shape(beta), dtype=np.double)
    p1 = np.zeros(N, dtype=np.double)
    for j in range(N):
        p1[j] = np.exp(np.dot(beta, data_x[j].T)) / (1 + np.exp(np.dot(beta, data_x[j].T)))
        jac = jac - (data_x[j]) * (data_y[j] - p1[j])
    return jac


def fun_hess(beta):
    hess = np.zeros((np.size(beta), np.size(beta)), dtype=np.double)
    p1 = np.zeros(N, dtype=np.double)
    for j in range(N):
        p1[j] = np.exp(np.dot(beta, data_x[j].T)) / (1 + np.exp(np.dot(beta, data_x[j].T)))
        hess += np.dot(data_x[j], data_x[j].T) * p1[j] * (1 - p1[j])
    return hess


def callback(xk):
    print(xk)


def line(beta, x):
    return 1 / beta[1] * (- beta[0] * x - beta[2])

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
# tol：迭代停止的精度。
# callback(xk)：每次迭代要回调的函数，需要有参数 xk
# options：其他选项
#     maxiter :  最大迭代次数
#     disp :  是否显示过程信息
# 以上参数更具体的介绍见官网相关页面。


if __name__ == '__main__':

    beta0 = np.array([[1., 1., 1.]])

    res = minimize(fun, beta0, callback=callback, tol=1.e-14,
                   options={'disp': True})

    print(res)

    n = 100
    # CLASS 1
    x_c1_test = np.random.randn(n, 2)
    x_c1_test = np.add(x_c1_test, [10, 10])
    ex_c1_test = np.concatenate((x_c1_test, np.ones((n, 1))), 1)  # 扩展权向量
    # CLASS 2
    x_c2_test = np.random.randn(n, 2)
    x_c2_test = np.add(x_c2_test, [2, 5])
    ex_c2_test = np.concatenate((x_c2_test, np.ones((n, 1))), 1)  # 扩展权向量

    data_x_test = np.concatenate((ex_c1_test, ex_c2_test), 0)

    x1 = x_c1_test[:, 0].T
    y1 = x_c1_test[:, 1].T
    x2 = x_c2_test[:, 0].T
    y2 = x_c2_test[:, 1].T

    # 主要用来创建等差数列，把0~10等分1000份的数列。
    X = np.linspace(0, 10, 1000)

    plt.plot(x1, y1, "b+", markersize=5)
    plt.plot(x2, y2, "r+", markersize=5)
    plt.plot(X, line(res.x, X), linestyle="-", color="black")

    plt.show()

    beta_x_hat = np.zeros(2 * n)
    y_hat = np.zeros(2 * n)

    for i in range(2 * n):
        beta_x_hat[i] = np.dot(res.x, data_x_test[i].T)
        y_hat[i] = np.exp(beta_x_hat[i]) / (1 + np.exp(beta_x_hat[i]))

    print(y_hat)
