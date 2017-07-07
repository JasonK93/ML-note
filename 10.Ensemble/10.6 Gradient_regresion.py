import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,cross_validation,ensemble

def load_data_regression():
    '''
    load the diabetes data for regression
    :return: train_data, test_data, train_value, test_value
    '''
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,diabetes.target,
    test_size=0.25,random_state=0)
def test_GradientBoostingRegressor(*data):
    '''
    test the method
    :param data: train_data, test_data, train_value, test_value
    :return:   None
    '''
    X_train,X_test,y_train,y_test=data
    regr=ensemble.GradientBoostingRegressor()
    regr.fit(X_train,y_train)
    print("Training score:%f"%regr.score(X_train,y_train))
    print("Testing score:%f"%regr.score(X_test,y_test))
def test_GradientBoostingRegressor_num(*data):
    '''
    test the performance with different n_estimators
    :param data:  train_data, test_data, train_value, test_value
    :return:   None
    '''
    X_train,X_test,y_train,y_test=data
    nums=np.arange(1,200,step=2)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for num in nums:
        regr=ensemble.GradientBoostingRegressor(n_estimators=num)
        regr.fit(X_train,y_train)
        training_scores.append(regr.score(X_train,y_train))
        testing_scores.append(regr.score(X_test,y_test))
    ax.plot(nums,training_scores,label="Training Score")
    ax.plot(nums,testing_scores,label="Testing Score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("GradientBoostingRegressor")
    plt.show()
def test_GradientBoostingRegressor_maxdepth(*data):
    '''
    test the performance with different max_depth
    :param data:   train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    maxdepths=np.arange(1,20)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for maxdepth in maxdepths:
        regr=ensemble.GradientBoostingRegressor(max_depth=maxdepth,max_leaf_nodes=None)
        regr.fit(X_train,y_train)
        training_scores.append(regr.score(X_train,y_train))
        testing_scores.append(regr.score(X_test,y_test))
    ax.plot(maxdepths,training_scores,label="Training Score")
    ax.plot(maxdepths,testing_scores,label="Testing Score")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(-1,1.05)
    plt.suptitle("GradientBoostingRegressor")
    plt.show()
def test_GradientBoostingRegressor_learning(*data):
    '''
    test the performance with different learning rate
    :param data:   train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    learnings=np.linspace(0.01,1.0)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for learning in learnings:
        regr=ensemble.GradientBoostingRegressor(learning_rate=learning)
        regr.fit(X_train,y_train)
        training_scores.append(regr.score(X_train,y_train))
        testing_scores.append(regr.score(X_test,y_test))
    ax.plot(learnings,training_scores,label="Training Score")
    ax.plot(learnings,testing_scores,label="Testing Score")
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(-1,1.05)
    plt.suptitle("GradientBoostingRegressor")
    plt.show()
def test_GradientBoostingRegressor_subsample(*data):
    '''
    test the performance with different subsample
    :param data:    train_data, test_data, train_value, test_value
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    subsamples=np.linspace(0.01,1.0,num=20)
    testing_scores=[]
    training_scores=[]
    for subsample in subsamples:
            regr=ensemble.GradientBoostingRegressor(subsample=subsample)
            regr.fit(X_train,y_train)
            training_scores.append(regr.score(X_train,y_train))
            testing_scores.append(regr.score(X_test,y_test))
    ax.plot(subsamples,training_scores,label="Training Score")
    ax.plot(subsamples,testing_scores,label="Training Score")
    ax.set_xlabel("subsample")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(-1,1.05)
    plt.suptitle("GradientBoostingRegressor")
    plt.show()
def test_GradientBoostingRegressor_loss(*data):
    '''
    test the performance with differnt loss function and alpha
    :param data:   train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    fig=plt.figure()
    nums=np.arange(1,200,step=2)
    ########## graph huber ######
    ax=fig.add_subplot(2,1,1)
    alphas=np.linspace(0.01,1.0,endpoint=False,num=5)
    for alpha in alphas:
            testing_scores=[]
            training_scores=[]
            for num in nums:
                    regr=ensemble.GradientBoostingRegressor(n_estimators=num,
					loss='huber',alpha=alpha)
                    regr.fit(X_train,y_train)
                    training_scores.append(regr.score(X_train,y_train))
                    testing_scores.append(regr.score(X_test,y_test))
            ax.plot(nums,training_scores,label="Training Score:alpha=%f"%alpha)
            ax.plot(nums,testing_scores,label="Testing Score:alpha=%f"%alpha)
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right",framealpha=0.4)
    ax.set_ylim(0,1.05)
    ax.set_title("loss=%huber")
    plt.suptitle("GradientBoostingRegressor")
    #### graph ls and lad
    ax=fig.add_subplot(2,1,2)
    for loss in ['ls','lad']:
        testing_scores=[]
        training_scores=[]
        for num in nums:
                regr=ensemble.GradientBoostingRegressor(n_estimators=num,loss=loss)
                regr.fit(X_train,y_train)
                training_scores.append(regr.score(X_train,y_train))
                testing_scores.append(regr.score(X_test,y_test))
        ax.plot(nums,training_scores,label="Training Score:loss=%s"%loss)
        ax.plot(nums,testing_scores,label="Testing Score:loss=%s"%loss)
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right",framealpha=0.4)
    ax.set_ylim(0,1.05)
    ax.set_title("loss=ls,lad")
    plt.suptitle("GradientBoostingRegressor")
    plt.show()
def test_GradientBoostingRegressor_max_features(*data):
    '''
    test the performance with different max_features
    :param data:  train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    max_features=np.linspace(0.01,1.0)
    testing_scores=[]
    training_scores=[]
    for features in max_features:
            regr=ensemble.GradientBoostingRegressor(max_features=features)
            regr.fit(X_train,y_train)
            training_scores.append(regr.score(X_train,y_train))
            testing_scores.append(regr.score(X_test,y_test))
    ax.plot(max_features,training_scores,label="Training Score")
    ax.plot(max_features,testing_scores,label="Training Score")
    ax.set_xlabel("max_features")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1.05)
    plt.suptitle("GradientBoostingRegressor")
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data_regression()
    test_GradientBoostingRegressor(X_train,X_test,y_train,y_test)
    test_GradientBoostingRegressor_num(X_train,X_test,y_train,y_test)
    test_GradientBoostingRegressor_maxdepth(X_train,X_test,y_train,y_test)
    test_GradientBoostingRegressor_learning(X_train,X_test,y_train,y_test)
    test_GradientBoostingRegressor_subsample(X_train,X_test,y_train,y_test)
    test_GradientBoostingRegressor_loss(X_train,X_test,y_train,y_test)
    test_GradientBoostingRegressor_max_features(X_train,X_test,y_train,y_test)
