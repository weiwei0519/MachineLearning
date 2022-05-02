# coding=UTF-8
# 使用SVD，对图像进行压缩

import numpy as np
import matplotlib.image as mping
import matplotlib.pyplot as plt
import matplotlib as mpl


def image_svd(n, pic):
    a, b, c = np.linalg.svd(pic)
    svd = np.zeros((a.shape[0], c.shape[1]))
    for i in range(0, n):
        svd[i, i] = b[i]
    img = np.matmul(a, svd)
    img = np.matmul(img, c)
    img[img >= 255] = 255
    img[0 >= img] = 0
    img = img.astype(np.uint8)
    return img


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = './img/example.jpg'
    img = mping.imread(path)
    print(img.shape)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    plt.figure(figsize=(50, 100))
    for i in range(1, 31):
        r_img = image_svd(i, r)
        g_img = image_svd(i, g)
        b_img = image_svd(i, b)
        pic = np.stack([r_img, g_img, b_img], axis=2)
        print(i)
        plt.subplot(5, 6, i)
        plt.title("图像的SVD分解，使用前 %d 个特征值" % (i))
        plt.axis('off')
        plt.imshow(pic)
    plt.suptitle("图像的SVD分解")
    plt.subplots_adjust()
    plt.show()
