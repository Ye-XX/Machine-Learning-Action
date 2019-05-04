import DecisionTree
import treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
featLabels = []
lensesTree = DecisionTree.createTree(lenses, lensesLabels, featLabels)
treePlotter.createPlot(lensesTree)
