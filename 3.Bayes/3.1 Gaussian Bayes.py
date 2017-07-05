from sklearn import datasets,cross_validation,naive_bayes
import  matplotlib.pyplot as plt

def load_data():
    '''
    reload the digits dataset from sklearn
    :return: train_data, test_data, train_value, test_value
    '''
    digits=datasets.load_digits()
    return cross_validation.train_test_split(digits.data,digits.target,
		test_size=0.25,random_state=0,stratify=digits.target)

def test_GaussianNB(*data):
    '''
    Test Gaussian NB
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    cls=naive_bayes.GaussianNB()
    cls.fit(X_train,y_train)
    print('Training Score: {0}' .format( cls.score(X_train,y_train)))
    print('Testing Score: {0}' .format( cls.score(X_test, y_test)))

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
    X_train,X_test,y_train,y_test=load_data()
    test_GaussianNB(X_train,X_test,y_train,y_test)