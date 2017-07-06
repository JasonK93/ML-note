import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,cross_validation,svm
def load_data_regression():
    '''
    load dataset for regression
    :return: train_data,test_data, train_target, test_target
    '''
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,diabetes.target,
		test_size=0.25,random_state=0)

def test_LinearSVR(*data):
    '''
    test Liner SVR
    :param data: train_data,test_data, train_target, test_target
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr=svm.LinearSVR()
    regr.fit(X_train,y_train)
    print('Coefficients:{0}, intercept {1}'.format(regr.coef_,regr.intercept_))
    print('Score: {0}' .format(regr.score(X_test, y_test)))
def test_LinearSVR_loss(*data):
    '''
    test SVr with different loss function
    :param data: train_data,test_data, train_target, test_target
    :return:
    '''
    X_train,X_test,y_train,y_test=data
    losses=['epsilon_insensitive','squared_epsilon_insensitive']
    for loss in losses:
        regr=svm.LinearSVR(loss=loss)
        regr.fit(X_train,y_train)
        print("loss：{0}".format(loss))
        print('Coefficients:{0}, intercept {1}'.format(regr.coef_,regr.intercept_))
        print('Score: {0}' .format(regr.score(X_test, y_test)))
def test_LinearSVR_epsilon(*data):
    '''
    test the performance with different epsilon
    :param data:  train_data,test_data, train_target, test_target
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    epsilons=np.logspace(-2,2)
    train_scores=[]
    test_scores=[]
    for  epsilon in  epsilons:
        regr=svm.LinearSVR(epsilon=epsilon,loss='squared_epsilon_insensitive')
        regr.fit(X_train,y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(epsilons,train_scores,label="Training score ",marker='+' )
    ax.plot(epsilons,test_scores,label= " Testing  score ",marker='o' )
    ax.set_title( "LinearSVR_epsilon ")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("score")
    ax.set_ylim(-1,1.05)
    ax.legend(loc="best",framealpha=0.5)
    plt.show()
def test_LinearSVR_C(*data):
    '''
    test the performance with different C
    :param data:  train_data,test_data, train_target, test_target
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    Cs=np.logspace(-1,2)
    train_scores=[]
    test_scores=[]
    for  C in  Cs:
        regr=svm.LinearSVR(epsilon=0.1,loss='squared_epsilon_insensitive',C=C)
        regr.fit(X_train,y_train)
        train_scores.append(regr.score(X_train, y_train))
        test_scores.append(regr.score(X_test, y_test))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,train_scores,label="Training score ",marker='+' )
    ax.plot(Cs,test_scores,label= " Testing  score ",marker='o' )
    ax.set_title( "LinearSVR_C ")
    ax.set_xscale("log")
    ax.set_xlabel(r"C")
    ax.set_ylabel("score")
    ax.set_ylim(-1,1.05)
    ax.legend(loc="best",framealpha=0.5)
    plt.show()
if __name__=="__main__":
    X_train,X_test,y_train,y_test=load_data_regression()
    test_LinearSVR(X_train,X_test,y_train,y_test)
    test_LinearSVR_loss(X_train,X_test,y_train,y_test)
    test_LinearSVR_epsilon(X_train,X_test,y_train,y_test)
    test_LinearSVR_C(X_train,X_test,y_train,y_test)