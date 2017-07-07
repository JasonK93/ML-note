import numpy as np
import matplotlib.pyplot as plt
from sklearn import  metrics
from sklearn import datasets
from sklearn.semi_supervised.label_propagation import LabelSpreading

def load_data():
    '''
    load data
    :return: data( have target), data_target, data( not have target)
    '''
    digits = datasets.load_digits()

    rng = np.random.RandomState(0)
    indices = np.arange(len(digits.data))
    rng.shuffle(indices)
    X = digits.data[indices]
    y = digits.target[indices]

    n_labeled_points = int(len(y)/10)
    unlabeled_indices = np.arange(len(y))[n_labeled_points:]

    return X,y,unlabeled_indices

def test_LabelSpreading(*data):
    '''
    test LabelSpreading
    :param data: data( have target), data_target, data( not have target)
    :return: None
    '''
    X,y,unlabeled_indices=data
    y_train=np.copy(y)
    y_train[unlabeled_indices]=-1
    clf=LabelSpreading(max_iter=100,kernel='rbf',gamma=0.1)
    clf.fit(X,y_train)

    predicted_labels = clf.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]
    print("Accuracy:%f"%metrics.accuracy_score(true_labels,predicted_labels))

def test_LabelSpreading_rbf(*data):
    '''
    test LabelSpreading with rbf kernel and different alpha, gamma
    :param data: data( have target), data_target, data( not have target)
    :return: None
    '''
    X,y,unlabeled_indices=data
    y_train=np.copy(y)
    y_train[unlabeled_indices]=-1

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    alphas=np.linspace(0.01,1,num=10,endpoint=True)
    gammas=np.logspace(-2,2,num=50)
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)

    for alpha,color in zip(alphas,colors):
        scores=[]
        for gamma in gammas:
            clf=LabelSpreading(max_iter=100,gamma=gamma,alpha=alpha,kernel='rbf')
            clf.fit(X,y_train)
            scores.append(clf.score(X[unlabeled_indices],y[unlabeled_indices]))
        ax.plot(gammas,scores,label=r"$\alpha=%s$"%alpha,color=color)


    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_xscale("log")
    ax.legend(loc="best")
    ax.set_title("LabelSpreading rbf kernel")
    plt.show()
def test_LabelSpreading_knn(*data):
    '''
    test LabelSpreading with knn kernel, and different alpha , n_neighbors
    :param data:  data( have target), data_target, data( not have target)
    :return:  None
    '''
    X,y,unlabeled_indices=data
    y_train=np.copy(y)
    y_train[unlabeled_indices]=-1

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    alphas=np.linspace(0.01,1,num=10,endpoint=True)
    Ks=[1,2,3,4,5,8,10,15,20,25,30,35,40,50]
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
        (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)

    for alpha,color in zip(alphas,colors):
        scores=[]
        for K in Ks:
            clf=LabelSpreading(kernel='knn',max_iter=100,n_neighbors=K,alpha=alpha)
            clf.fit(X,y_train)
            scores.append(clf.score(X[unlabeled_indices],y[unlabeled_indices]))
        ax.plot(Ks,scores,label=r"$\alpha=%s$"%alpha,color=color)


    ax.set_xlabel(r"$k$")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("LabelSpreading knn kernel")
    plt.show()
if __name__=='__main__':
    data=load_data()
    test_LabelSpreading(*data)
    test_LabelSpreading_rbf(*data)
    test_LabelSpreading_knn(*data)