import numpy as np
import operator
from os import listdir

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

# 将32*32的图像转化为1*1024的向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# 手写数字分类测试
def handWritingClassTest():
    hwLabels = []  # 测试集的Labels
    trainingFileList = listdir('trainingDigits')  # 返回trainingDigits目录下的文件名
    m = len(trainingFileList)                     # 返回文件夹下文件的个数
    trainingMat = np.zeros((m, 1024))             # 初始化测试集的训练矩阵Mat,每行存储一个图像
    # 从文件名中解析出训练集的类别
    for i in range(m):
        fileNameStr = trainingFileList[i]               # 获取文件名
        classNumber = int(fileNameStr.split('_')[0])    # 获得分类的数字('_'前面的第一个数字）
        hwLabels.append(classNumber)                    # 将获得的类别添加到hwLabels中
        # 将每个文件的1*1024数据存储到trainingMat矩阵中
        trainingMat[i:] = img2vector('trainingDigits/%s' % (fileNameStr))
    """
    对testDigits目录中的文件执行相似的操作，不同之处是我们并不将这个目录下的文件载入矩阵中，
    而是使用classify0()函数测试该目录下的每个文件。由于文件中的值已经在0和1之间，
    并不需要使用autoNorm()函数。
    """
    testFileList = listdir('testDigits')        # 返回testDigits目录下的文件名
    errorCount = 0.0                            # 错误检测计数
    mTest = len(testFileList)                   # 测试数据的数量
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        # 获得测试集的1*1024向量，用于训练
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # 获得预测结果
        classFierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类返回的结果为：%d,\t真实结果为：%d" % (classFierResult, classNumber))
        if (classFierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount*100 / mTest))

if __name__ == '__main__':
    handWritingClassTest()
