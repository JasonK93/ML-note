# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,cross_validation

def load_data():
    '''
    load for the dataset
        return:
                1 array for the classification problem.
                train_data, test_data, train_value, test_value
    '''
    iris=datasets.load_iris() # Use the IRIS data. This data has 3 class and 150 examples
    X_train=iris.data
    y_train=iris.target
    return cross_validation.train_test_split(X_train, y_train,test_size=0.25,
		random_state=0,stratify=y_train)
def test_LogisticRegression(*data):
    '''
    test of LR
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train, y_train)
    print('Coefficients: {0}, intercept {1}'.format(regr.coef_,regr.intercept_))
    print('Score: {0}' .format(regr.score(X_test, y_test)))
def test_LogisticRegression_multinomial(*data):
    '''
    Test with different multi_class
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
    regr.fit(X_train, y_train)
    print('Coefficients: {0}, intercept {1}'.format(regr.coef_,regr.intercept_))
    print('Score: {0}' .format(regr.score(X_test, y_test)))
def test_LogisticRegression_C(*data):
    '''
    test score with different C
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    Cs=np.logspace(-2,4,num=100)
    scores=[]
    for C in Cs:
        regr = linear_model.LogisticRegression(C=C)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    ## graph
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,scores)
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("LogisticRegression")
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    test_LogisticRegression(X_train,X_test,y_train,y_test)
    test_LogisticRegression_multinomial(X_train,X_test,y_train,y_test)
    test_LogisticRegression_C(X_train,X_test,y_train,y_test)