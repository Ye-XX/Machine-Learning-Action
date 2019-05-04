from math import log
import operator
import pickle
import treePlotter


# 计算香农熵
def calcShannonEnt(dataSet):
    """
    【作用】计算给定数据集的经验熵（香农熵）
    :param dataSet: 数据集
    :return: 经验熵（香农熵）
    """
    numEntries = len(dataSet)               # 返回数据集的行数
    labelCounts = {}                        # 每个标签出现的次数
    for featVec in dataSet:                 # 对每组特征向量进行统计
        currentLabel = featVec[-1]          # 提取标签的信息
        if currentLabel not in labelCounts.keys():      # 如果标签没有放入统计次数的字典，则添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1      # Label计数
    shannonEnt = 0.0
    for key in labelCounts:                 # 计算香农熵
        prob = float(labelCounts[key]) / numEntries     # 选择该标签的概率
        shannonEnt -= prob * log(prob, 2)   # 利用公式进行计算
    return shannonEnt


# 创建数据集
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet, labels


# 划分数据集
def splitDataSet(dataSet, feature, valus):
    """
    :param dataSet: 待划分的数据集
    :param feature: 划分数据集的特征
    :param valus: 需要返回的特征值
    :return:划分后的数据集
    """
    retDataSet = []                     # 创建返回的数据集列表
    for featVec in dataSet:
        if featVec[feature] == valus:
            reducedFeatVec = featVec[:feature]              # 去除掉feature特征（存在疑问？？？？？）
            reducedFeatVec.extend(featVec[feature + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选取最优特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1               # 特征数量
    baseEntropy = calcShannonEnt(dataSet)           # 计算数据集的香农熵
    bestInfoGain = 0.0                              # 信息增益
    bestFeature = -1                                # 最优特征值的索引值
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 获取数据集中所有第i个特征值
        uniqueVals = set(featList)                  # 创建集合(set)数据集，数据集中的元素互不相同
        newEntropy = 0.0                            # 经验条件熵
        for value in uniqueVals:                    # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 统计出现次数最多的标签
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                      # 统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 根据字典的值降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树
def createTree(dataSet, labels, featLabels):
    """
    :param dataSet: 训练数据集
    :param labels: 分类属性标签
    :param featLabels: 存储选择的最优特征标签
    :return: 决策树
    """
    classList = [example[-1] for example in dataSet]        # 获取分类标签
    if classList.count(classList[0]) == len(classList):     # 如果所以分类标签都相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:               # 遍历完所以特征时返回出现次数最多的特征
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}                                # 根据最优特征的标签生成树
    del(labels[bestFeat])                                      # 删除已使用的特征标签
    featValues = [example[bestFeat] for example in dataSet]     # 获取数据集中所以最优特征的属性值
    uniqueVals = set(featValues)                                # 去掉重复的属性值
    for value in uniqueVals:                                    # 遍历特征，创建决策树
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
    return myTree


# 使用决策树进行分类
def classify(inputTree, featLabels, testVec):
    """
    :param inputTree: 已生成的决策树
    :param featLabels: 存储选择的最优特征标签
    :param testVec: 测试数据列表，顺序对应最优特征标签
    :return: classLabel——分类标签
    """
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)      # 将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 存储决策树
def storeTree(inputTree, filename):
    """
    :param inputTree: 已生成的决策树
    :param filename: 决策树的存储文件名
    :return: 无
    """
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


# 读取决策树
def gradTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    treePlotter.createPlot(myTree)
    testVec = [0, 1]
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print("放贷")
    if result == 'no':
        print("不放贷")
