import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, cross_validation

def create_regression_data(n):
    '''
    generate data
    :param n: the space of data set
    :return: train_data, test_data, train_value, test_value
    '''
    X =5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    y[::5] += 1 * (0.5 - np.random.rand(int(n/5)))
    return cross_validation.train_test_split(X, y,test_size=0.25,random_state=0)

def test_KNeighborsRegressor(*data):
    '''
    test the KNN regressor
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr=neighbors.KNeighborsRegressor()
    regr.fit(X_train,y_train)
    print("Training Score:{0}".format(regr.score(X_train,y_train)))
    print("Testing Score:{0}".format(regr.score(X_test,y_test)))
def test_KNeighborsRegressor_k_w(*data):
    '''
    test the performance with different n_neighbors and weights
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    Ks=np.linspace(1,y_train.size,num=100,endpoint=False,dtype='int')
    weights=['uniform','distance']

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ### graph
    for weight in weights:
        training_scores=[]
        testing_scores=[]
        for K in Ks:
            regr=neighbors.KNeighborsRegressor(weights=weight,n_neighbors=K)
            regr.fit(X_train,y_train)
            testing_scores.append(regr.score(X_test,y_test))
            training_scores.append(regr.score(X_train,y_train))
        ax.plot(Ks,testing_scores,label="testing score:weight={0}".format(weight))
        ax.plot(Ks,training_scores,label="training score:weight={0}".format(weight))
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNeighborsRegressor")
    plt.show()
def test_KNeighborsRegressor_k_p(*data):
    '''
    test the performance with different n_neighbors and p
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    Ks=np.linspace(1,y_train.size,endpoint=False,dtype='int')
    Ps=[1,2,10]

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ### graph
    for P in Ps:
        training_scores=[]
        testing_scores=[]
        for K in Ks:
            regr=neighbors.KNeighborsRegressor(p=P,n_neighbors=K)
            regr.fit(X_train,y_train)
            testing_scores.append(regr.score(X_test,y_test))
            training_scores.append(regr.score(X_train,y_train))
        ax.plot(Ks,testing_scores,label="testing score:p={0}".format(P))
        ax.plot(Ks,training_scores,label="training score:p={0}".format(P))
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNeighborsRegressor")
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test=create_regression_data(1000)
    test_KNeighborsRegressor(X_train,X_test,y_train,y_test)
    test_KNeighborsRegressor_k_w(X_train,X_test,y_train,y_test)
    test_KNeighborsRegressor_k_p(X_train,X_test,y_train,y_test)