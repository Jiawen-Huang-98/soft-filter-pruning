# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：Cifar数据转换.py
@时间：2022/4/1  10:49
@文档说明:
"""

import numpy as np
import pickle as pkl
import imageio


# 将cifar10数据转化为图片+标签格式

# 定义反序列函数
def unpickle(file):
    fo = open(file, 'rb')
    dict = pkl.load(fo, encoding = 'bytes')  # 以二进制的方式加载
    fo.close()
    return dict


# 转换train数据集
# for j in range(1, 6):
#     dataName = "D:/Xdown资源/Xdown下载任务/cifar-10-batches-py/data_batch_" + str(j)
#     Xtr = unpickle(dataName)
#     print(dataName + " is loading...")
#
#     for i in range(0, 10000):
#         img = np.reshape(Xtr[b'data'][i], (3, 32, 32))
#         img = img.transpose(1, 2, 0)  # 三维转置
#         picName = 'D:/Xdown资源/Xdown下载任务/cifar-10-batches-py/data_batch_/train/' + str(Xtr[b'labels'][i]) + '/' + str(
#             i + (j - 1) * 10000) + '.jpg'
#         imageio.imwrite(picName, img)
#     print(dataName + " loaded.")

print("test_batch is loading...")

# 转换test数据集
testXtr = unpickle("D:/Xdown资源/Xdown下载任务/cifar-10-batches-py/test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'D:/Xdown资源/Xdown下载任务/cifar-10-batches-py/data_batch_test/' + str(testXtr[b'labels'][i]) + '_' + str(i) + '.jpg'
    imageio.imwrite(picName, img)
print("test_batch loaded.")
