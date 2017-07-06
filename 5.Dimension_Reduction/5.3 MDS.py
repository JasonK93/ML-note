import numpy as np
import matplotlib.pyplot as plt
from sklearn import   datasets,manifold

def load_data():
    '''
    load the iris data
    :return: train_data, train_value
    '''
    iris=datasets.load_iris()
    return  iris.data,iris.target

def test_MDS(*data):
    '''
    test MDS method
    :param data: train_data, train_value
    :return: None
    '''
    X,y=data
    for n in [4,3,2,1]:
        mds=manifold.MDS(n_components=n)
        mds.fit(X)
        print('stress(n_components={0}) : {1}'.format (n, str(mds.stress_)))
def plot_MDS(*data):
    '''
    graph after MDS
    :param data: train_data, train_value
    :return: None
    '''
    X,y=data
    mds=manifold.MDS(n_components=2)
    X_r=mds.fit_transform(X)

    ### graph
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)
    for label ,color in zip( np.unique(y),colors):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="target= {0}".format(label),color=color)

    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title("MDS")
    plt.show()
if __name__=='__main__':
    X,y=load_data()
    test_MDS(X,y)
    plot_MDS(X,y)