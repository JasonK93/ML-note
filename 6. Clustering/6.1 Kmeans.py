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



def test_Kmeans(*data):
    '''
    test the Kmeans
    :param data: data, target
    :return: None
    '''
    X,labels_true=data
    clst=cluster.KMeans()
    clst.fit(X)
    predicted_labels=clst.predict(X)
    print("ARI:{0}".format( adjusted_rand_score(labels_true,predicted_labels)))
    print("Sum center distance {0}".format(clst.inertia_))
def test_Kmeans_nclusters(*data):
    '''
    test the performance with different n_clusters
    :param data: data, target
    :return: None
    '''
    X,labels_true=data
    nums=range(1,50)
    ARIs=[]
    Distances=[]
    for num in nums:
        clst=cluster.KMeans(n_clusters=num)
        clst.fit(X)
        predicted_labels=clst.predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        Distances.append(clst.inertia_)

    ## graph
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs,marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances,marker='o')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("inertia_")
    fig.suptitle("KMeans")
    plt.show()
def test_Kmeans_n_init(*data):
    '''
    test the performance with different n_init and init paramter
    :param data: data, target
    :return: None
    '''
    X,labels_true=data
    nums=range(1,50)
    ## graph
    fig=plt.figure()

    ARIs_k=[]
    Distances_k=[]
    ARIs_r=[]
    Distances_r=[]
    for num in nums:
            clst=cluster.KMeans(n_init=num,init='k-means++')
            clst.fit(X)
            predicted_labels=clst.predict(X)
            ARIs_k.append(adjusted_rand_score(labels_true,predicted_labels))
            Distances_k.append(clst.inertia_)

            clst=cluster.KMeans(n_init=num,init='random')
            clst.fit(X)
            predicted_labels=clst.predict(X)
            ARIs_r.append(adjusted_rand_score(labels_true,predicted_labels))
            Distances_r.append(clst.inertia_)

    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs_k,marker="+",label="k-means++")
    ax.plot(nums,ARIs_r,marker="+",label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("ARI")
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances_k,marker='o',label="k-means++")
    ax.plot(nums,Distances_r,marker='o',label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("inertia_")
    ax.legend(loc='best')

    fig.suptitle("KMeans")
    plt.show()

if __name__=='__main__':
    centers=[[1,1],[2,2],[1,2],[10,20]]
    X,labels_true=create_data(centers,1000,0.5)
    plot_data(X,labels_true)
    test_Kmeans(X,labels_true)
    test_Kmeans_nclusters(X,labels_true)
    test_Kmeans_n_init(X,labels_true)
