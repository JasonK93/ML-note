
   # 简介

决策树的功能很强大，是一种有监督的学习方法。决策树既可以用来解决回归问题，也可以解决分类问题。

   # 原理

在特征空间上执行递归的二元分割. 。有节点和有向边组成。

   # 步骤

特征选择；决策树生成；决策树剪枝。

特征选择根据：熵，基尼系数，方差等因素决定。生成决策树的方法有很多，典型的有ID3，和C4.5。 ID3 采用的信息增益作为度量。C4.5采用信息增益比。树剪枝简化了模型，并且某种程度上减少了过拟合的发生。同时树剪枝也是预测误差和数据复杂度之间的一个折中。

# 实战代码：GitHub

1.决策树分类：

https://github.com/JasonK93/ML-note/blob/master/2.decision%20tree(DT)/2.1%20Decision%20Tree-Classifier.py

2.决策树回归：

https://github.com/JasonK93/ML-note/blob/master/2.decision%20tree(DT)/2.2%20Decision%20Tree-%20Regression.py

  #  决策图

决策树生成后可以对相关规则进行可视化，使用函数export_graphviz()
