
# 综述

贝叶斯分类原理是通过对某对象的先验概率，利用贝叶斯公式计算出后验概率，再选取最大的概率的事件作为分类对象。

# 分类器

1.高斯分类器GaussianNB：条件概率分布满足高斯分布

https://github.com/JasonK93/ML-note/blob/master/3.Bayes/3.1%20Gaussian%20Bayes.py

2.多项式贝叶斯分类器（MultinomialNB）：条件概率满足多项式分布

https://github.com/JasonK93/ML-note/blob/master/3.Bayes/3.2%20Multinomial%20NB.py

3.伯努利贝叶斯分类器（BernouliNB):条件概率满足伯努利分布

https://github.com/JasonK93/ML-note/blob/master/3.Bayes/3.3%20Bernoulli%20NB.py

# Partial_fit

贝叶斯可以处理大规模数据，当完整的训练集无法放入内存中的时候，可以动态的增加数据来进行使用—-online classifier。将一个大数据集分割成数个数据集分块训练。
