
#    综述

聚类算法是非监督学习，目的在于对于一些没有target的数据集，进行分类的算法。这是一种探索性分析的方法，用来分析数据的内在特点，寻找数据的分布规律。

 #   聚类的有效性指标

聚类的有效性指标主要有两种，外部指标，内部指标。

外部指标：由聚类结果与某个参考模型比较获得。1.Jaccard 系数；2.FM指数；3.Rand 指数；4.ARI指数。

内部指标：由考察聚类结果直接获得。1.DB指数；2.Dunn指数。

  #  度量距离

欧几里得距离，曼哈顿距离，汉明距离，VDM距离等等

   # 原型聚类

常用的原型聚类由，K均值聚类，高斯混合聚类等等。K均值的目标函数是最小均方误差。高斯混合聚类，假设聚类服从高斯分布。

 #   密度聚类

Density-based clustering 假设聚类结构能够通过样本分布的紧密程度来确定。常用的算法由，DBSCAN。

  #  层次聚类

hierarchical clustering  可在不同层上对数据集进行划分。形成类似树一样的聚类结构。

 #   EM算法

期望最大算法，是一种迭代方法，主要用于含有隐变量的概率模型的参数估计。其中主要分两步，E为求期望，M为求极大。在混合高斯聚类等方法中有应用。

  #  现实任务中的聚类要求：

1.可伸缩性：数据量的变化不影响聚类结果的准确度。2.不同类型数据的处理能力要求。3.适应不同类簇形状的混合聚类要求。4.初始化参数的敏感性的解决要求。5.算法的抗噪能力。6.增量聚类的实现。7.对输入次序的敏感度把握要求。8.高维数据的处理能力要求。9.结果的可读性，可视化性，可解释性，与可应用性。

#实战代码：GitHub

1.Kmeans:

https://github.com/JasonK93/ML-note/blob/master/6.%20Clustering/6.1%20Kmeans.py

2.DBSCAN:

https://github.com/JasonK93/ML-note/blob/master/6.%20Clustering/6.2%20DBSCAN.py

3. Agglomerative Clustering:

https://github.com/JasonK93/ML-note/blob/master/6.%20Clustering/6.3%20Agglomerative%20Clustering.py

4.GaussianMixture:

https://github.com/JasonK93/ML-note/blob/master/6.%20Clustering/6.4%20GaussianMixture.py
