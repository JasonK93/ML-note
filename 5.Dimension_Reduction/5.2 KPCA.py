import numpy as np
import matplotlib.pyplot as plt
from sklearn import   datasets,decomposition

def load_data():
    '''
    load the iris data
    :return: train_data, train_value
    '''
    iris=datasets.load_iris()# 使用 scikit-learn 自带的 iris 数据集
    return  iris.data,iris.target

def test_KPCA(*data):
    '''
    test the KPCA method
    :param data: train_data, train_value
    :return: None
    '''
    X,y=data
    kernels=['linear','poly','rbf','sigmoid']
    for kernel in kernels:
        kpca=decomposition.KernelPCA(n_components=None,kernel=kernel) # Use 4 different kernel
        kpca.fit(X)
        print('kernel={0} --> lambdas: {1}'.format (kernel,kpca.lambdas_))
def plot_KPCA(*data):
    '''
    graph after KPCA
    :param data: train_data, train_value
    :return: None
    '''
    X,y=data
    kernels=['linear','poly','rbf','sigmoid']
    fig=plt.figure()
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)

    for i,kernel in enumerate(kernels):
        kpca=decomposition.KernelPCA(n_components=2,kernel=kernel)
        kpca.fit(X)
        X_r=kpca.transform(X)
        ax=fig.add_subplot(2,2,i+1)
        for label ,color in zip( np.unique(y),colors):
            position=y==label
            ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label,
            color=color)
        ax.set_xlabel("X[0]")
        ax.set_ylabel("X[1]")
        ax.legend(loc="best")
        ax.set_title("kernel={0}".format(kernel))
    plt.suptitle("KPCA")
    plt.show()
def plot_KPCA_poly(*data):
    '''
    graph after KPCA with poly kernel
    :param data: train_data, train_value
    :return: None
    '''
    X,y=data
    fig=plt.figure()
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)
    Params=[(3,1,1),(3,10,1),(3,1,10),(3,10,10),(10,1,1),(10,10,1),(10,1,10),(10,10,10)] # parameter of poly
            # p ， gamma ， r ）
            # p ：3，10
            # gamma  ：1，10
            # r ：1，10
            # 8 combination
    for i,(p,gamma,r) in enumerate(Params):
        kpca=decomposition.KernelPCA(n_components=2,kernel='poly'
        ,gamma=gamma,degree=p,coef0=r)
        kpca.fit(X)
        X_r=kpca.transform(X)
        ax=fig.add_subplot(2,4,i+1)
        for label ,color in zip( np.unique(y),colors):
            position=y==label
            ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"%label,
            color=color)
        ax.set_xlabel("X[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("X[1]")
        ax.legend(loc="best")
        ax.set_title(r"$ ({0} (x \cdot z+1)+{1})^{{2}}$".format(gamma,r,p))
    plt.suptitle("KPCA-Poly")
    plt.show()
def plot_KPCA_rbf(*data):
    '''
    graph with kernel of rbf
    :param data: train_data, train_value
    :return: None
    '''
    X,y=data
    fig=plt.figure()
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)
    Gammas=[0.5,1,4,10]
    for i,gamma in enumerate(Gammas):
        kpca=decomposition.KernelPCA(n_components=2,kernel='rbf',gamma=gamma)
        kpca.fit(X)
        X_r=kpca.transform(X)
        ax=fig.add_subplot(2,2,i+1)
        for label ,color in zip( np.unique(y),colors):
            position=y==label
            ax.scatter(X_r[position,0],X_r[position,1],label="target= {0}".format(label),
            color=color)
        ax.set_xlabel("X[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("X[1]")
        ax.legend(loc="best")
        ax.set_title(r"$\exp(-{0}||x-z||^2)$".format(gamma))
    plt.suptitle("KPCA-rbf")
    plt.show()
def plot_KPCA_sigmoid(*data):
    '''
    graph with sigmoid kernel
    :param data: train_data, train_value
    :return: None
    '''
    X,y=data
    fig=plt.figure()
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)
    Params=[(0.01,0.1),(0.01,0.2),(0.1,0.1),(0.1,0.2),(0.2,0.1),(0.2,0.2)]# parameter of sigmoid kernel
        # gamma,coef0
        # gamma ： 0.01，0.1，0.2
        # coef0 ： 0.1,0.2
        # 6 combination
    for i,(gamma,r) in enumerate(Params):
        kpca=decomposition.KernelPCA(n_components=2,kernel='sigmoid',gamma=gamma,coef0=r)
        kpca.fit(X)
        X_r=kpca.transform(X)
        ax=fig.add_subplot(3,2,i+1)
        for label ,color in zip( np.unique(y),colors):
            position=y==label
            ax.scatter(X_r[position,0],X_r[position,1],label="target= {0}".format(label),
            color=color)
        ax.set_xlabel("X[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("X[1]")
        ax.legend(loc="best")
        ax.set_title(r"$\tanh({0}(x\cdot z)+{1})$".format(gamma,r))
    plt.suptitle("KPCA-sigmoid")
    plt.show()
if __name__=='__main__':
    X,y=load_data()
    test_KPCA(X,y)
    plot_KPCA(X,y)
    plot_KPCA_poly(X,y)
    plot_KPCA_rbf(X,y)
    plot_KPCA_sigmoid(X,y)