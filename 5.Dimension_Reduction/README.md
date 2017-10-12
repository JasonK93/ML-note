
#    综述

针对数据特征的处理，从而避免维度灾难，并且减少噪音数据特征的影响，提高精度。

 #   PCA

主成分分析法，是一种维度上的压缩变换。但是由于是无监督的压缩，很多的时候是将开始的特征进行了线性组合，从而生成了新的不能合理解释的新的特征。

https://github.com/JasonK93/ML-note/blob/master/5.Dimension_Reduction/5.1%20PCA.py

 #   SVD

奇异值分解降维。该方法等价于PCA主成分分析，核心都是求解XX（T）的特征值以及对应的特征向量。

 #   KPCA

核主成分分析法。由于主成分分析法是线性的降维，并不能满足现实任务中的要求，所以需要非线性映射的降维。所以有了基于核技术的降维方法，核主成分分析。

https://github.com/JasonK93/ML-note/blob/master/5.Dimension_Reduction/5.2%20KPCA.py

 #   流形学习降维

流形学习是一种借鉴了拓扑流形概念的降维方法，是一种非线性的降维方法。其特点在于，构造的局部邻域不同，利用这些邻域结构构造全局的低维嵌入方法不同。

#    MDS

多维缩放降维，要求原始空间中的样本之间的距离在低维空间中得到保持。

https://github.com/JasonK93/ML-note/blob/master/5.Dimension_Reduction/5.3%20MDS.py

 #   Isomap

等度量映射降维，利用流形在局部上与欧几里得空间同胚的性质，找到每个点在低维流形上的邻近点近邻连接图。计算最短路径问题。利用MDS方法获得低维空间。

https://github.com/JasonK93/ML-note/blob/master/5.Dimension_Reduction/5.4%20Isomap.py

 #   LLE

局部线性嵌入降维的主要目标是，降维的同时保证邻域内样本的线性关系。

 

https://github.com/JasonK93/ML-note/blob/master/5.Dimension_Reduction/5.5%20LLE.py
