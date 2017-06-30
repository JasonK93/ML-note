import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,cross_validation


def load_data():
    '''
    load for the dataset
    return:
            1 array for the regression problem.
            train_data, test_data, train_value, test_value
    '''

    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,diabetes.target,
		test_size=0.25,random_state=0)

def test_Ridge(*data):
    '''
    test the ridge analysis
    :param data:  train_data, test_data, train_value, test_value
    :return: None
    '''

    X_train,X_test,y_train,y_test=data
    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    print('Coefficients:{0}, intercept {1}'.format(regr.coef_,regr.intercept_))
    print("Residual sum of squares: {0}".format(np.mean((regr.predict(X_test) - y_test) ** 2)))
    print('Score: {0}' .format(regr.score(X_test, y_test)))
def test_Ridge_alpha(*data):
    '''
    test the score with different alpha param
    :param data:  train_data, test_data, train_value, test_value
    :return: None
    '''

    X_train,X_test,y_train,y_test=data
    alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    '''
    actually, smaller alpha means a better score. But consider the calculation power, we need to trade off.
    '''
    scores=[]
    for i,alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    ## graph
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Ridge")
    plt.show()
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    test_Ridge(X_train,X_test,y_train,y_test)
    test_Ridge_alpha(X_train,X_test,y_train,y_test)