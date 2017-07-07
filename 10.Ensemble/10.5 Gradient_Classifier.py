import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,cross_validation,ensemble

def load_data_classification():
    '''
    load the digits data set for classification
    :return:  train_data, test_data, train_value, test_value
    '''
    digits=datasets.load_digits()
    return cross_validation.train_test_split(digits.data,digits.target,
    test_size=0.25,random_state=0,stratify=digits.target)
def test_GradientBoostingClassifier(*data):
    '''
    test the method
    :param data:   train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    clf=ensemble.GradientBoostingClassifier()
    clf.fit(X_train,y_train)
    print("Traing Score:%f"%clf.score(X_train,y_train))
    print("Testing Score:%f"%clf.score(X_test,y_test))
def test_GradientBoostingClassifier_num(*data):
    '''
    test the performance with different n_estimators
    :param data:    train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    nums=np.arange(1,100,step=2)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for num in nums:
        clf=ensemble.GradientBoostingClassifier(n_estimators=num)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    ax.plot(nums,training_scores,label="Training Score")
    ax.plot(nums,testing_scores,label="Testing Score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()
def test_GradientBoostingClassifier_maxdepth(*data):
    '''
    test the performance with different max_depth
    :param data:     train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    maxdepths=np.arange(1,20)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for maxdepth in maxdepths:
        clf=ensemble.GradientBoostingClassifier(max_depth=maxdepth,max_leaf_nodes=None)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    ax.plot(maxdepths,training_scores,label="Training Score")
    ax.plot(maxdepths,testing_scores,label="Testing Score")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()
def test_GradientBoostingClassifier_learning(*data):
    '''
    test the performance with different learning rate
    :param data:     train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    learnings=np.linspace(0.01,1.0)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for learning in learnings:
        clf=ensemble.GradientBoostingClassifier(learning_rate=learning)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    ax.plot(learnings,training_scores,label="Training Score")
    ax.plot(learnings,testing_scores,label="Testing Score")
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()
def test_GradientBoostingClassifier_subsample(*data):
    '''
    test the performance with different subsample
    :param data:    train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    subsamples=np.linspace(0.01,1.0)
    testing_scores=[]
    training_scores=[]
    for subsample in subsamples:
            clf=ensemble.GradientBoostingClassifier(subsample=subsample)
            clf.fit(X_train,y_train)
            training_scores.append(clf.score(X_train,y_train))
            testing_scores.append(clf.score(X_test,y_test))
    ax.plot(subsamples,training_scores,label="Training Score")
    ax.plot(subsamples,testing_scores,label="Training Score")
    ax.set_xlabel("subsample")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()
def test_GradientBoostingClassifier_max_features(*data):
    '''
    test the performance with different max_features
    :param data:     train_data, test_data, train_value, test_value
    :return:   None
    '''
    X_train,X_test,y_train,y_test=data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    max_features=np.linspace(0.01,1.0)
    testing_scores=[]
    training_scores=[]
    for features in max_features:
            clf=ensemble.GradientBoostingClassifier(max_features=features)
            clf.fit(X_train,y_train)
            training_scores.append(clf.score(X_train,y_train))
            testing_scores.append(clf.score(X_test,y_test))
    ax.plot(max_features,training_scores,label="Training Score")
    ax.plot(max_features,testing_scores,label="Training Score")
    ax.set_xlabel("max_features")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("GradientBoostingClassifier")
    plt.show()
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data_classification()
    test_GradientBoostingClassifier(X_train,X_test,y_train,y_test)
    test_GradientBoostingClassifier_num(X_train,X_test,y_train,y_test)
    test_GradientBoostingClassifier_maxdepth(X_train,X_test,y_train,y_test)
    test_GradientBoostingClassifier_learning(X_train,X_test,y_train,y_test)
    test_GradientBoostingClassifier_subsample(X_train,X_test,y_train,y_test)
    test_GradientBoostingClassifier_max_features(X_train,X_test,y_train,y_test)
