from sklearn.feature_selection import  SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import  load_digits,load_diabetes
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def test_SelectFromModel():
    '''
    test the method of SelectFromModel
    :return: None
    '''
    digits=load_digits()
    X=digits.data
    y=digits.target
    estimator=LinearSVC(penalty='l1',dual=False)
    selector=SelectFromModel(estimator=estimator,threshold='mean')
    selector.fit(X,y)
    selector.transform(X)
    print("Threshold %s"%selector.threshold_)
    print("Support is %s"%selector.get_support(indices=True))
def test_Lasso(*data):
    '''
    test the correlation between alpha and sparse condition
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X,y=data
    alphas=np.logspace(-2,2)
    zeros=[]
    for alpha in alphas:
        regr=Lasso(alpha=alpha)
        regr.fit(X,y)
        num=0
        for ele in regr.coef_:
            if abs(ele) < 1e-5:num+=1
        zeros.append(num)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,zeros)
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_ylim(0,X.shape[1]+1)
    ax.set_ylabel("zeros in coef")
    ax.set_title("Sparsity In Lasso")
    plt.show()
def test_LinearSVC(*data):
    '''
    test the correlation between C and sparse condition
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X,y=data
    Cs=np.logspace(-2,2)
    zeros=[]
    for C in Cs:
        clf=LinearSVC(C=C,penalty='l1',dual=False)
        clf.fit(X,y)

        num=0
        for row in clf.coef_:
            for ele in row:
                if abs(ele) < 1e-5:num+=1
        zeros.append(num)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,zeros)
    ax.set_xlabel("C")
    ax.set_xscale("log")
    ax.set_ylabel("zeros in coef")
    ax.set_title("Sparsity In SVM")
    plt.show()
if __name__=='__main__':
    test_SelectFromModel()
    data=load_diabetes()
    test_Lasso(data.data,data.target)
    data=load_digits()
    test_LinearSVC(data.data,data.target)