# -*- coding: utf-8 -*-
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

def test_ElasticNet(*data):
    '''
    test for Elastic Net
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.ElasticNet()
    regr.fit(X_train, y_train)
    print('Coefficients:{0}, intercept {1}'.format(regr.coef_, regr.intercept_))
    print("Residual sum of squares: {0}".format(np.mean((regr.predict(X_test) - y_test) ** 2)))
    print('Score: {0}'.format(regr.score(X_test, y_test)))
def test_ElasticNet_alpha_rho(*data):
    '''
    test score with different alpha and l1_ratio
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    alphas=np.logspace(-2,2)
    rhos=np.linspace(0.01,1)
    scores=[]
    for alpha in alphas:
            for rho in rhos:
                regr = linear_model.ElasticNet(alpha=alpha,l1_ratio=rho)
                regr.fit(X_train, y_train)
                scores.append(regr.score(X_test, y_test))
    ## graph
    alphas, rhos = np.meshgrid(alphas, rhos)
    scores=np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D # this part works well in py3
    from matplotlib import cm
    fig=plt.figure()
    ax=Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    test_ElasticNet(X_train,X_test,y_train,y_test)
    test_ElasticNet_alpha_rho(X_train,X_test,y_train,y_test)