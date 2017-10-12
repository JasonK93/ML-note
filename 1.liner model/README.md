
 #   线性模型的一般表达形式：

f(x) = W*X + b

其中， W= （w1, w2, w3, …, wn)T, 称之为权重向量。 权重向量直接的表现了各个特征在预测中的重要性。广义线性模型可以理解为是一个N维的线性模型。

线性回归，是一种监督性学习。结果是连续的，可以理解为回归分析； 结果是离散的， 可以理解为是分类问题。

  #  比较常见的相关模型有：

岭回归， Lasso回归， Elastic Net， 逻辑回归， 线性判别分析等。

   # 线性模型的损失函数一般为平房损失函数：

（预测值-真实值）的平方和；（目标是使损失函数最小。）

   # 对特征使用归一化（ Feature Scaling）：

对特征进行处理，使之特征空间更加圆润适合训练学习。优点：1）提升模型的收敛速度；2）提神模型精度

   # 正则化：

对于由于变量过多而导致的过拟合问题，有两种主要的解决方法。其一是降低维度，同时会避免唯独灾难。其二就是正则化，保留了所有的变量的同时改变了他们的数量级以改变模型性能。

   # 逻辑回归

对于逻辑回归而言，多了一个激活函数。

   # 线性判别分析（LDA）

LDA的思想是： 在训练时，将训练样本投影到一条直线上，是的同类的点尽可能接近，异类的点尽可能远离。在预测时，根据投影位置来判断类别。

    LAD的目标函数是：使同类点的方差尽可能小–J1，异类中心点距离尽可能大–J2。即使J1/J2尽可能小。

# 实战代码：GitHub

1.线性模型：

https://github.com/JasonK93/ML-note/blob/master/1.liner%20model/1.basic%20liner%20model.py

2.岭分析

https://github.com/JasonK93/ML-note/blob/master/1.liner%20model/2.ridge%20analysis.py

3.Lasso分析

https://github.com/JasonK93/ML-note/blob/master/1.liner%20model/3.Lasso%20regression.py

4.Elastic NET

https://github.com/JasonK93/ML-note/blob/master/1.liner%20model/4.%20ElasticNet.py

5.逻辑回归

https://github.com/JasonK93/ML-note/blob/master/1.liner%20model/5.logistic%20regression.py

6. LDA

https://github.com/JasonK93/ML-note/blob/master/1.liner%20model/6.LDA.py
