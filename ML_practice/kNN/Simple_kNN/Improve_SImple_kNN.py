# k-近邻算法的简单实现
# -*- coding: UTF-8 -*-
# Time：2019-04-18
# Author：WuYe

from numpy import *
import collections
""""
1、import numpy：如果使用numpy的属性都需要在前面加上numpy
2、from numpy import *：则不需要在调用前加numpy
如调用random模块，第一种为numpy.random，第二种为random
"""

# 创建数据集和标签
def createDataSet():
    group = array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    labels = ['爱情片', '爱情片', '爱情片', '动作片', '动作片', '动作片']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    Function：kNN的分类器
    :param inX: 用于分类的输入向量（测试集）
    :param dataSet: 输入的训练样本集（训练集）
    :param labels: 分类标签向量
    :param k: 用于选择最近相似的数目
    :return: inX的分类结果
    注：标签向量的元素数目和矩阵dataSet的行数相同
    """
    # 计算距离
    dist = sum((inX - dataSet) ** 2, axis=1) ** 0.5
    # k个最近的标签
    k_labels = [labels[index] for index in dist.argsort()[0: k]]
    # 出现次数最多的标签即为最终类别
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label

if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [101, 5]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    # 打印分类结果
    print(test_class)