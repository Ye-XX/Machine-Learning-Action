# k-近邻算法的简单实现
# -*- coding: UTF-8 -*-
# Time：2019-04-18
# Author：WuYe

from numpy import *
import operator     # 运算符模块
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
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # ===================== 欧氏距离计算公式 =====================
    # 行向量方向上重复inX共dataSetSize次(纵向),在列向量方向上重复inX共1次(横向)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 计算完所有点之间的距离后，对数据按照从小到大的次序排序。
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 确定前k个距离最小元素所在的分类
        voteIlabel = labels[sortedDistIndices[i]]
        """dict.get(key,default=None),字典的get()方法,
        返回指定键的值,如果值不在字典中返回默认值。
        """
        # 将classCount字典分解为元组列表。计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    """
    python3中用items()替换python2中的iteritems()
    key=operator.itemgetter(1)根据字典的值进行排序
    key=operator.itemgetter(0)根据字典的键进行排序
    reverse降序排序字典
    """
    # 按照第二个元素的次序对元组进行排序（此处的排序为逆序）
    # (python2)sortedClassCount = sorted(classCount.itertems(), key=operator.itemgetter(1), reverse=True)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [101, 5]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    # 打印分类结果
    print(test_class)