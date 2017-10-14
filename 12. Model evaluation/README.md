#  综述

模型的评估在机器学习中扮演着很重要的角色，用于分别什么才是好的预测模型。机器学习一般包含两个方面，原型设计和应用。原型设计方面会通过验证和离线评估来选择一个较好的模型。评估方法一般有在线评估和离线评估等。在线评估一般是在应用阶段使用新生成的数据来进行评估并更新模型的过程。

 #   离线评估，在线评估

离线评估中，我们一般会使用到 准确率(accuracy), 精确率(precision), 召回率(recall)。而在线评估有用户生命周期价值（Customer Lifetime Value), 广告点击率( Click Through Rate), 用户流失率(Customer Churn Rate) 等等。

  #  损失函数

损失函数一般用于度量错误的程度。 常用的有：0-1损失函数， 平方损失函数，绝对损失函数， 对数损失函数。 风险函数定义为损失函数的期望，所以学习的目标也可以是风险函数最小的模型。

   # 模型评估

度量因素：训练误差，测试误差。根据这两个因素可以推论是否有过拟合或者欠拟合的情况。评估方法常用有：1.留出法：也可以说是三分法（ train data, valid data, test data). 2 交叉验证法(Cross- Validation) 3. 留一法( Leave-One-Out) 4. 自助法(Boostrapping)

   # 性能度量

准确率，错误率， 混淆矩阵，precision, recall, P-R curve(  Precision-Recall 曲线， 被包住的性能好）， ROC曲线

实战代码：GitHub：

https://github.com/JasonK93/ML-note/tree/master/12.%20Model%20evaluation




