# -*- coding: utf-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import  datasets
from sklearn import cross_validation
import matplotlib.pyplot as plt
def load_data():
    '''
    load iris data from sk-learn. this data has 150 samples and 3 class.
        return:
                1 array for the classification problem.
                train_data, test_data, train_value, test_value
    '''
    iris=datasets.load_iris()
    X_train=iris.data
    y_train=iris.target
    return cross_validation.train_test_split(X_train, y_train,test_size=0.25,
		random_state=0,stratify=y_train)
def test_DecisionTreeClassifier(*data):
    '''
    test decision tree
    :param data: train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    print("Training score: {0}".format(clf.score(X_train,y_train)))
    print("Testing score: {0}".format(clf.score(X_test,y_test)))
def test_DecisionTreeClassifier_criterion(*data):
    '''
    test the performance with different criterion
    :param data: train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    criterions=['gini','entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train, y_train)
        print("criterion:{0}".format(criterion))
        print("Training score: {0}".format(clf.score(X_train,y_train)))
        print("Testing score: {0}".format(clf.score(X_test,y_test)))
def test_DecisionTreeClassifier_splitter(*data):
    '''
    test the performance with different splitters
    :param data: train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    splitters=['best','random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train, y_train)
        print("splitter: {0}".format(splitter))
        print("Training score:{0}".format(clf.score(X_train,y_train)))
        print("Testing score: {0}".format(clf.score(X_test,y_test)))
def test_DecisionTreeClassifier_depth(*data,maxdepth):
    '''
    test the score with different max_depth
    :param data: train_data, test_data, train_value, test_value
    :param maxdepth: an integer
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    depths=np.arange(1,maxdepth)
    training_scores=[]
    testing_scores=[]
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))

    ## graph
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label="traing score",marker='o')
    ax.plot(depths,testing_scores,label="testing score",marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5,loc='best')
    plt.show()
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    test_DecisionTreeClassifier(X_train,X_test,y_train,y_test)
    test_DecisionTreeClassifier_criterion(X_train,X_test,y_train,y_test)
    test_DecisionTreeClassifier_splitter(X_train,X_test,y_train,y_test)
    test_DecisionTreeClassifier_depth(X_train,X_test,y_train,y_test,maxdepth=100)