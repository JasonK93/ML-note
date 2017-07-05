from sklearn import datasets,cross_validation,naive_bayes
import  matplotlib.pyplot as plt
import numpy as np

def load_data():
    '''
    reload the digits dataset from sklearn
    :return: train_data, test_data, train_value, test_value
    '''
    digits=datasets.load_digits()
    return cross_validation.train_test_split(digits.data,digits.target,
		test_size=0.25,random_state=0,stratify=digits.target)

def test_BernoulliNB(*data):
    '''
    test BernoulliNB
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    cls=naive_bayes.BernoulliNB()
    cls.fit(X_train,y_train)
    print('Training Score: {0}'.format(cls.score(X_train,y_train)))
    print('Testing Score: {0}'.format(cls.score(X_test, y_test)))
def test_BernoulliNB_alpha(*data):
    '''
    test the performance with different alpha
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    alphas=np.logspace(-2,5,num=200)
    train_scores=[]
    test_scores=[]
    for alpha in alphas:
        cls=naive_bayes.BernoulliNB(alpha=alpha)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test, y_test))

    ## graph
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,train_scores,label="Training Score")
    ax.plot(alphas,test_scores,label="Testing Score")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.0)
    ax.set_title("BernoulliNB")
    ax.set_xscale("log")
    ax.legend(loc="best")
    plt.show()
def test_BernoulliNB_binarize(*data):
    '''
    test the performance with different binarize
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    min_x=min(np.min(X_train.ravel()),np.min(X_test.ravel()))-0.1
    max_x=max(np.max(X_train.ravel()),np.max(X_test.ravel()))+0.1
    binarizes=np.linspace(min_x,max_x,endpoint=True,num=100)
    train_scores=[]
    test_scores=[]
    for binarize in binarizes:
        cls=naive_bayes.BernoulliNB(binarize=binarize)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test, y_test))

    ## graph
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(binarizes,train_scores,label="Training Score")
    ax.plot(binarizes,test_scores,label="Testing Score")
    ax.set_xlabel("binarize")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.0)
    ax.set_xlim(min_x-1,max_x+1)
    ax.set_title("BernoulliNB")
    ax.legend(loc="best")
    plt.show()
def show_digits():
    '''
    graph the first 25 samples in the data set
    :return: None
    '''
    digits=datasets.load_digits()
    fig=plt.figure()
    print("vector from images 0:",digits.data[0])
    for i in range(25):
        ax=fig.add_subplot(5,5,i+1)
        ax.imshow(digits.images[i],cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

if __name__=='__main__':
    show_digits()
    X_train, X_test, y_train, y_test = load_data()
    test_BernoulliNB(X_train, X_test, y_train, y_test)
    test_BernoulliNB_alpha(X_train, X_test, y_train, y_test)
    test_BernoulliNB_binarize(X_train, X_test, y_train, y_test)