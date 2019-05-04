import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# 获取叶子节点的数目
def getNumLeafs(myTree):
    numLeafs = 0                        # 初始化叶子节点的数目
    """
    python3中myTree.keys()返回的是dict_keys,不再是list,所以不能使用
    myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    """
    # firstStr = list(myTree.key()[0])
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]       # 获取下一组字典
    for key in secondDict.keys():       # 测试结点是否为字典，若不是字典，则代表此节点为叶子节点
        if type(secondDict[key]).__name__ == 'dict':
            # 如果子节点是字典类型，则该节点也是一个判断节点，需要递归调用getNumLeafs()函数。
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取决策树的层数（计算遍历过程中遇到判断节点的个数）
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':     # 若测试结点字典类型，则此节点为判断节点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1       # （原文）琢磨一下
            # thisDepth += 1      # 觉得应该是这样
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 绘制带箭头的注解
def plotNode(nodeTex, centerPt, parentPt, nodeType):
    """
    :param nodeTex: 节点名
    :param centerPt: 文本位置
    :param parentPt: 标注的箭头位置
    :param nodeType: 节点格式
    :return:
    """
    arrow_args = dict(arrowstyle="<-")                  # 定义箭头格式
    font = FontProperties(fname=r"C:\Windows\Fonts\STKAITI.TTF", size=14)   # 设置中文字体
    createPlot.ax1.annotate(nodeTex, xy=parentPt, xycoords='axes fraction',
                           xytext=centerPt, textcoords='axes fraction',
                           va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


# 标注有向边属性值
def plotMidText(cntrPt, parentPt, txtString):
    """
    :param cntrPt:用于计算标注位置
    :param parentPt:用于计算标注位置
    :param txtString:标注的内容
    :return:无
    """
    # 在父子节点间填充文本信息
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
    """
    :param myTree: 决策树（字典）
    :param parentPt: 标注内容
    :param nodeTxt: 结点名
    :return: 无
    【注】树的宽度用于计算放置判断节点的位置，主要的计算原则是将它放在所有叶子节点的中间，而不仅仅是它子节点的中间。
    同时我们使用两个全局变量plotTree.xOff和plotTree.yOff追踪已经绘制的节点位置，以及放置下一个节点的恰当位置。
    通过计算树包含的所有叶子节点数，划分图形的宽度，从而计算得到当前节点的中心位置，也就是说，我们按照叶子节点的数目
    将x轴划分为若干部分。
    """
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")      # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")            # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)      # 获取决策树叶节点的个数，决定了树的宽度
    depth = getTreeDepth(myTree)        # 获取决策树的深度，决定了树的高度
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)   # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                  # 标记子节点属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)      # 绘制节点
    secondDict = myTree[firstStr]                           # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD     # 自顶向下绘制图形，因此需要依次递减y坐标值
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


# 创建绘制面板
def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")  # 绘制图形
    fig.clf()  # 清空绘图区
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)     # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))        # 获取决策树叶节点数目
    plotTree.totalD = float(getTreeDepth(inTree))       # 获取决策树节点
    plotTree.xOff = -0.5/plotTree.totalW                # x偏移
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')                    # 绘制决策树
    plt.show()                                          # 显示绘制结果

