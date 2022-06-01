# coding=UTF-8
# 模型评估

#混淆矩阵评估
def mixed_mat(Y_pred, Y_test):
    # 混淆矩阵
    TP = 0  # 真正例
    FN = 0  # 假反例
    FP = 0  # 假正例
    TN = 0  # 真反例

    # 通过后验概率，预测Y ~ （0,1）
    for i in range(Y_test.shape[0]):
        if Y_pred[i] == 1:
            if Y_pred[i] == Y_test[i]:
                TP += 1
            else:
                FP += 1
        else:
            if Y_pred[i] == Y_test[i]:
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
    return P, R, F1