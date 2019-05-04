# k-近邻算法的简单实现
# -*- coding: UTF-8 -*-
# Time：2019-04-18
# Author：WuYe

import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import operator

# 处理数据格式
def file2matrix(filename):
    """
    【作用】
        打开并解析文件，对数据进行分类
    【参数说明】
        returnMat：特征矩阵
        classLabelVector：分类Label向量
    """
    fr = open(filename, 'r', encoding='utf-8')
    # 读取文件所有内容
    arrayOLines = fr.readlines()
    # 针对有BOM的UTF-8文本，应该去掉BOM，否则后面会引发错误。
    arrayOLines[0] = arrayOLines[0].lstrip('\ufeff')
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines, 3))
    # 返回的分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0

    for line in arrayOLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

# 数据可视化
def showData(datingDataMat, datingLabels):
    """
    【作用】可视化数据
    :param datingDataMat: 特征矩阵
    :param datingLabels: 分类标签
    :return: 数据图
    """
    # 设置汉字格式
    font = FontProperties(fname="C:\Windows\Fonts\simkai.ttf", size=14)
    # 将fit图画分隔成1行1列，不共享x轴和y轴，fig图画的大小为（13,8）
    # 当nrow=2, nclos=2时，代表fig图画被分成四个领域，axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('blue')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图，以datingDataMat矩阵的第一列（飞行常客里程），
    # 第二列（玩游戏时间）数据画散点数据，散点大小为15，透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=0.5)
    # 设置标题title、x轴label、y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间', FontProperties=font)
    # 画出标题title、x轴label、y轴label
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图，以datingDataMat矩阵的第二列（飞行常客里程），
    # 第三列（冰淇淋）数据画散点数据，散点大小为15，透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=0.5)
    # 设置标题title、x轴label、y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰淇淋公升数', FontProperties=font)
    # 画出标题title、x轴label、y轴label
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图，以datingDataMat矩阵的第一列（玩游戏时间），
    # 第三列（冰淇淋）数据画散点数据，散点大小为15，透明度为0.5
    axs[1][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=0.5)
    # 设置标题title、x轴label、y轴label
    axs2_title_text = axs[1][0].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰淇淋公升数', FontProperties=font)
    # 画出标题title、x轴label、y轴label
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='blue', marker='.', markersize=6, label='samllDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()

# 数据归一化
def autoNorm(dataSet):
    """"
    【目的】
        数据归一化
    【参数说明】
        dataSet：特征矩阵
    【返回】
        normDataSet：归一化后的特征矩阵
        range：数据范围
        minVals：数据最小值
    【注】将任意取值范围的特征值转化为0到1区间内的值：newValue = (oldValue - min) / (max - min)
    """
    # 获得数据的最小值和最大值,dataSet.min(0)中的参数0使得函数可以从列中选取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # shape(dataSet)返回dataSet的矩阵 行列数
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回dataSet的 行数
    m = dataSet.shape[0]
    # 特征值矩阵有1000×3个值，而minVals和range的值都为1×3。为了解决这个问题，
    # 我们使用NumPy库中tile()函数将变量内容复制成输入矩阵同样大小的矩阵。
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals

# 分类器
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
    dataSetSize = dataSet.shape[0]           # numpy函数shape[0]返回dataSet的行数
    # ===================== 欧氏距离计算公式 =====================
    # 行向量方向上重复inX共dataSetSize次(纵向),在列向量方向上重复inX共1次(横向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)         # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()    # 计算完所有点之间的距离后，对数据按照从小到大的次序排序。
    classCount = {}     # 定一个记录类别次数的字典
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
    # print(sortedClassCount)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

# 分类器测试
def datingClassTest():
    """
    【作用】
        分类器的测试函数
    【打印】
        错误率
    """
    filename = "datingTestSet.txt"
    # 取所有数据的百分之十
    hoRatio = 0.10
    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]
    # 获取测试数据的数量
    numTestVecs = int(m * hoRatio)
    # 计数分类的错误次数
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify0(normMat[i, :], normMat[numTestVecs: m, :], datingLabels[numTestVecs:m], 3)
        print("分类器的分类结果：%d，真实类别为：%d" % (classfierResult, datingLabels[i]))
        if (classfierResult != datingLabels[i]):
            errorCount += 1
    print("出错次数：%d" % errorCount)
    print("分类器的错误率：%f %%" % (errorCount/float(numTestVecs)*100))

# 预测函数
def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有点喜欢', '非常喜欢']
    # 三维特征用户输入
    precentTats = float(input("玩视屏游戏所消耗的时间百分百："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("买周消费的冰淇淋公升数："))
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    # 训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 生成Numpy数组，测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals) / ranges
    # 返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    # 打印结果
    print("你可能%s这个人" % (resultList[classifierResult - 1]))

if __name__ == '__main__':
    # filename = "datingTestSet.txt"
    # returnMat, classLabelVector = file2matrix(filename)
    # showData(returnMat, classLabelVector)
    classifyPerson()
    # datingClassTest()

