from sklearn import  cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import adjusted_rand_score


def create_data(centers,num=100,std=0.7):
    '''
    generate data
    :param centers: dimension of centre
    :param num: number of sample
    :param std: std of each cluster
    :return: data, target
    '''
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return  X,labels_true
def plot_data(*data):
    '''
    graph the dataset
    :param data: data, target
    :return: None
    '''
    X,labels_true=data
    labels=np.unique(labels_true)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors='rgbyckm'
    for i,label in enumerate(labels):
        position=labels_true==label
        ax.scatter(X[position,0],X[position,1],label="cluster {0}".format(label),
		color=colors[i%len(colors)])

    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title("data")
    plt.show()

def test_AgglomerativeClustering(*data):
    '''
    test AGG method
    :param data: data, target
    :return: None
    '''
    X,labels_true=data
    clst=cluster.AgglomerativeClustering()
    predicted_labels=clst.fit_predict(X)
    print("ARI:{0}".format(adjusted_rand_score(labels_true,predicted_labels)))
def test_AgglomerativeClustering_nclusters(*data):
    '''
    test the performance with different n_clusters
    :param data:  data, target
    :return: None
    '''
    X,labels_true=data
    nums=range(1,50)
    ARIs=[]
    for num in nums:
        clst=cluster.AgglomerativeClustering(n_clusters=num)
        predicted_labels=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))

    ## graph
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(nums,ARIs,marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    fig.suptitle("AgglomerativeClustering")
    plt.show()
def test_AgglomerativeClustering_linkage(*data):
    '''
    test the performance with different linkages
    :param data:  data, target
    :return: None
    '''
    X,labels_true=data
    nums=range(1,50)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    linkages=['ward','complete','average']
    markers="+o*"
    for i, linkage in enumerate(linkages):
        ARIs=[]
        for num in nums:
            clst=cluster.AgglomerativeClustering(n_clusters=num,linkage=linkage)
            predicted_labels=clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        ax.plot(nums,ARIs,marker=markers[i],label="linkage:{0}".format(linkage))

    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    fig.suptitle("AgglomerativeClustering")
    plt.show()

if __name__=='__main__':
    centers=[[1,1],[2,2],[1,2],[10,20]]
    X,labels_true=create_data(centers,1000,0.5)
    plot_data(X,labels_true)
    test_AgglomerativeClustering(X,labels_true)
    test_AgglomerativeClustering_nclusters(X,labels_true)
    test_AgglomerativeClustering_linkage(X,labels_true)


