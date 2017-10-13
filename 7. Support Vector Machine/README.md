
  #  综述

支持向量机（Support Vector Machine) 在使用核技术之后，是可以进行非线性分类的。模型的基本定义，是使得在空间中分类的间隔最大化。分割是超平面分割，目标函数是是满足KKT条件下的最大值或者倒数最小值。

   # 涉及知识点：

决策函数， KKT条件， 对偶问题，拉格朗日函数， 惩罚参数，

   # 常用核函数：

多项式核函数， 高斯核函数， sigmoid 核函数等等。

   # 支持向量机回归

Support Vector Regression( SVR) , 损失函数是一范数， 但是一般会设置参数，当距离大于参数时，才考虑损失函数。

   # SVM优缺点：

优点：可以解决非线性的优化问题。避免了神经网络的结构选择， 核局部极小点问题。

缺点：确实数据敏感，对于非线性问题，很依赖于核函数的选择，没有通用的解法。主流算法的时间复杂度是O（n2)，所以在大规模数据下的计算需要庞大的计算量。同时结果对超参数的依赖程度很大。（比如RBF核的超参数， gamma核惩罚项C）

# 实战代码：GitHub

1.SVM 线性分类-SVC

https://github.com/JasonK93/ML-note/blob/master/7.%20Support%20Vector%20Machine/7.1%20SVM-liner_SVC.py

2. SVM非线性分类-SVC

https://github.com/JasonK93/ML-note/blob/master/7.%20Support%20Vector%20Machine/7.2%20SVM-unliner_SVC.py

3. SVM线性回归-SVR

https://github.com/JasonK93/ML-note/blob/master/7.%20Support%20Vector%20Machine/7.3%20liner_SVR.py

4.SVM非线性回归-SVR

https://github.com/JasonK93/ML-note/blob/master/7.%20Support%20Vector%20Machine/7.4%20unliner_SVR.py

