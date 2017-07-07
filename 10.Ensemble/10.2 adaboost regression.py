import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,cross_validation,ensemble

def load_data_regression():
    '''
    load the date set for regression (diabetes)
    :return: train_data, test_data, train_value, test_value
    '''
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,diabetes.target,
    test_size=0.25,random_state=0)

def test_AdaBoostRegressor(*data):
    '''
    test the regression with different number of regression model
    :param data: train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    regr=ensemble.AdaBoostRegressor()
    regr.fit(X_train,y_train)
    ## graph
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    estimators_num=len(regr.estimators_)
    X=range(1,estimators_num+1)
    ax.plot(list(X),list(regr.staged_score(X_train,y_train)),label="Traing score")
    ax.plot(list(X),list(regr.staged_score(X_test,y_test)),label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostRegressor")
    plt.show()
def test_AdaBoostRegressor_base_regr(*data):
    '''
    test the regression with different number of model and regression method
    :param data:  train_data, test_data, train_value, test_value
    :return: None
    '''
    from sklearn.svm import  LinearSVR
    X_train,X_test,y_train,y_test=data
    fig=plt.figure()
    regrs=[ensemble.AdaBoostRegressor(),
		ensemble.AdaBoostRegressor(base_estimator=LinearSVR(epsilon=0.01,C=100))]
    labels=["Decision Tree Regressor","Linear SVM Regressor"]
    for i ,regr in enumerate(regrs):
        ax=fig.add_subplot(2,1,i+1)
        regr.fit(X_train,y_train)
        ## graph
        estimators_num=len(regr.estimators_)
        X=range(1,estimators_num+1)
        ax.plot(list(X),list(regr.staged_score(X_train,y_train)),label="Traing score")
        ax.plot(list(X),list(regr.staged_score(X_test,y_test)),label="Testing score")
        ax.set_xlabel("estimator num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(-1,1)
        ax.set_title("Base_Estimator:%s"%labels[i])
    plt.suptitle("AdaBoostRegressor")
    plt.show()
def test_AdaBoostRegressor_learning_rate(*data):
    '''
    test the performance with different learning rate
    :param data:   train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    learning_rates=np.linspace(0.01,1)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    traing_scores=[]
    testing_scores=[]
    for learning_rate in learning_rates:
        regr=ensemble.AdaBoostRegressor(learning_rate=learning_rate,n_estimators=500)
        regr.fit(X_train,y_train)
        traing_scores.append(regr.score(X_train,y_train))
        testing_scores.append(regr.score(X_test,y_test))
    ax.plot(learning_rates,traing_scores,label="Traing score")
    ax.plot(learning_rates,testing_scores,label="Testing score")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostRegressor")
    plt.show()
def test_AdaBoostRegressor_loss(*data):
    '''
    test the method with different loss function
    :param data:    train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    losses=['linear','square','exponential']
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for i ,loss in enumerate(losses):
        regr=ensemble.AdaBoostRegressor(loss=loss,n_estimators=30)
        regr.fit(X_train,y_train)
        ## graph
        estimators_num=len(regr.estimators_)
        X=range(1,estimators_num+1)
        ax.plot(list(X),list(regr.staged_score(X_train,y_train)),
			label="Traing score:loss=%s"%loss)
        ax.plot(list(X),list(regr.staged_score(X_test,y_test)),
			label="Testing score:loss=%s"%loss)
        ax.set_xlabel("estimator num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(-1,1)
    plt.suptitle("AdaBoostRegressor")
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data_regression()
    test_AdaBoostRegressor(X_train,X_test,y_train,y_test)
    test_AdaBoostRegressor_base_regr(X_train,X_test,y_train,y_test)
    test_AdaBoostRegressor_learning_rate(X_train,X_test,y_train,y_test)
    test_AdaBoostRegressor_loss(X_train,X_test,y_train,y_test)